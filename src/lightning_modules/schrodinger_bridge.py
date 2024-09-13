import torch
from typing import Tuple, Any, Callable
from torch import Tensor
from src.lightning_modules.baselightningmodule import BaseLightningModule
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

class StandardSchrodingerBridge(BaseLightningModule):
    def __init__(
        self,
        forward_model : torch.nn.Module,
        backward_model : torch.nn.Module,
        max_gamma : float = None,
        min_gamma : float = None,
        num_steps : int = None,
        strict_gammas : bool = True,
        patience: int = 100,
        lr : float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore = ["forward_model", "backward_model"])
        self.automatic_optimization : bool = False
        self.training_backward : bool = True

        assert max_gamma >= min_gamma, f"{max_gamma = } must be greater than {min_gamma = }"
        gammas = torch.linspace(min_gamma, 2 * max_gamma - min_gamma, num_steps)
        gammas[-(num_steps // 2) :] = reversed(gammas[: num_steps // 2])
        gammas = torch.cat([torch.tensor([0]), gammas]) # the first gamma is 0. This gamma is never used, since indexing starts at 1
        self.gammas = gammas
        
        if strict_gammas:
            gammas_sum = gammas.sum().item()
            assert abs(1 - gammas_sum) < 1e-3, f"The sum of gammas must be equal to 1, but got {gammas_sum = }"
        
        self.forward_model : Callable[[Tensor, Tensor], Tensor] = forward_model
        self.backward_model : Callable[[Tensor, Tensor], Tensor] = backward_model
        
        self.mse : Callable[[Tensor, Tensor], Tensor] = torch.nn.MSELoss()        
        self.val_losses : list = []
        self.DSB_iteration : int = 0

    def has_converged(self) -> bool:
        losses, patience = self.val_losses, self.hparams.patience

        if len(losses) < patience + 1:
            return False
        
        min_loss = min(losses[:-patience])
        return all([l > min_loss for l in losses[-patience:]])
    
    def on_train_batch_start(self, batch : Tensor, batch_idx : int) -> None:
        """
        here we check if the model has converged
        if the model has converged we do the following:
        - switch the training direction
        - reset the val_losses
        - reset the learning rate of the optimizer
        - reset the learning rate scheduler
        we also return -1 to skip the rest of the epoch if converged
        
        """

        if self.has_converged():
            if not self.training_backward: 
                self.DSB_iteration += 1
                
            self.training_backward = not self.training_backward
            self.val_losses = []
            
            # resetting the learning rate
            for optimizer in self.optimizers():
                for pg in optimizer.param_groups:
                    pg['lr'] = self.hparams.lr
                    
            # resetting the learning rate scheduler
            for scheduler in self.lr_schedulers():
                scheduler.num_bad_epochs = 0
            
            return -1
    
    def k_to_tensor(self, k : int, size : Tuple[int]) -> Tensor:
        return torch.full((size, ), k, dtype = torch.float32, device = self.device)
    
    @torch.no_grad()
    def go_forward(self, xk : Tensor, k : int) -> Tensor:
        """
        Get x_{k + 1} | x_k
        
        Args:
            xk (Tensor) : the current point
            k (int) : the current step
            
        Returns:
            Tensor : the next point
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def go_backward(self, xk_plus_one : Tensor, k_plus_one : int) -> Tensor:
        """
        Get x_{k} | x_{k + 1}
        
        Args:
            xk (Tensor) : the current point
            k (int) : the current step
            
        Returns:
            Tensor : the previous point
        """
        raise NotImplementedError
    
    def _forward_loss(self, xk : Tensor, k : int, xN : Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute the loss for the forward model

        Args:
            xk (Tensor) : the current point
            k (int) : the current step
            xN (Tensor) : the end point
            
        ### Returns:
        - Tuple[Tensor, Tensor]
            - the loss
            - the next point
        """
        raise NotImplementedError
    
    def _backward_loss(self, xk : Tensor, k : int, x0 : Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute the loss for the backward model
        
        Args:
            xk (Tensor) : the current point
            k (int) : the current step
            x0 (Tensor) : the start point
            
        Returns:
            Tuple[Tensor, Tensor] : the loss and the previous point
        """
        raise NotImplementedError
    
    def sample(self, x_start : Tensor, forward : bool = True, return_trajectory : bool = False) -> Tensor:
        """
        Sample from the forward or backward model starting from x_start
        
        Args:
            x_start (Tensor) : the starting point
            forward (bool) : whether to sample from the forward model
            return_trajectory (bool) : whether to return the trajectory. The trajectory is a tensor of shape (num_steps, *x_start.size())
            
        Returns:
            Tensor : the sampled point
        """
        
        trajectory = torch.zeros(self.hparams.num_steps + 1, *x_start.size())
        
        if forward:
            xk = x_start
            trajectory[0] = xk
            for k in range(self.hparams.num_steps): # 0, 1, 2, ..., num_steps - 1
                xk_plus_one = self.go_forward(xk, k)
                trajectory[k + 1] = xk_plus_one
                xk = xk_plus_one

        else:
            xk_plus_one = x_start
            trajectory[-1] = xk_plus_one
            for k in reversed(range(self.hparams.num_steps)): # num_steps - 1, num_steps - 2, ..., 1, 0
                xk = self.go_backward(xk_plus_one, k + 1)
                trajectory[k] = xk
                xk_plus_one = xk
            
        return trajectory if return_trajectory else xk
    
    def _train_backward(self, x0 : Tensor, validating : bool = False) -> float:
        """
        Given the start point x0, train the backward model
        
        Args:
            x0 (Tensor) : the start point
            
        Returns:
            losses (dict) : the average backward and forward losses
        """
        backward_opt, _ = self.optimizers()
        batch_losses = torch.zeros(self.hparams.num_steps)
        xk = x0.clone()
        
        for i, k in enumerate(range(self.hparams.num_steps)):
            loss, xk_plus_one = self._backward_loss(xk, k, x0)
            
            if not validating:
                backward_opt.zero_grad()
                self.manual_backward(loss)
                clip_grad_norm_(self.backward_model.parameters(), 1)
                backward_opt.step()
                
            xk = xk_plus_one
            batch_losses[i] = loss.item()
            
        avg_loss = batch_losses.mean().item()
        
        return avg_loss
    
    def _train_forward(self, xN : Tensor, validating : bool = False) -> float:
        """
        Given the end point xN, train the forward model

        Args:
            xN (Tensor) : the end point

        Returns:
            losses (dict) : the average backward and forward losses

        """
        _, forward_opt = self.optimizers()
        batch_losses = torch.zeros(self.hparams.num_steps)
        xk_plus_one = xN.clone()
        
        for i, k in enumerate(reversed(range(self.hparams.num_steps))):
            loss, xk = self._forward_loss(xk_plus_one, k + 1, xN)
            
            if not validating:
                forward_opt.zero_grad()
                self.manual_backward(loss)
                clip_grad_norm_(self.forward_model.parameters(), 1)
                forward_opt.step()
                
            xk_plus_one = xk
            batch_losses[i] = loss.item()
            
        avg_loss = batch_losses.mean().item()
        
        return avg_loss
    
    def training_step(self, batch : Tensor, batch_idx : int) -> None:
        x0, xN = batch
        
        if self.training_backward:
            avg_loss = self._train_backward(x0)
            self.log("backward_loss/train", avg_loss, prog_bar = True)
        else:
            avg_loss = self._train_forward(xN)
            self.log("forward_loss/train", avg_loss, prog_bar = True)

        self.log("DSB_iteration", self.DSB_iteration, prog_bar = True)
    
    @torch.no_grad()
    def validation_step(self, batch : Tensor, batch_idx : int, dataloader_idx : int) -> None:        
        if dataloader_idx == 0 and self.training_backward:
            avg_loss = self._train_backward(batch, validating = True)
            self.log("backward_loss/val", avg_loss, prog_bar = True, add_dataloader_idx=False)

        elif dataloader_idx == 1 and not self.training_backward:
            avg_loss = self._train_forward(batch, validating = True)
            self.log("forward_loss/val", avg_loss, prog_bar = True, add_dataloader_idx=False)

    def on_validation_epoch_end(self) -> None:
        backward_scheduler, forward_scheduler = self.lr_schedulers()
        metrics = self.trainer.callback_metrics

        if len(metrics) == 0:
            return
        
        if self.training_backward:
            val_loss = metrics["backward_loss/val"].item()
            backward_scheduler.step(val_loss)
        else:
            val_loss = metrics["forward_loss/val"].item()
            forward_scheduler.step(val_loss)

        self.val_losses.append(val_loss)
        
    def configure_optimizers(self):
        backward_opt = AdamW(self.backward_model.parameters(), lr = self.hparams.lr)
        forward_opt = AdamW(self.forward_model.parameters(), lr = self.hparams.lr)
        
        # convergence will happen after 'patience' validation steps with no improvement
        # therefore reduce the lr after 'patience // 2' validation steps with no improvement
        lr_args = {
            'patience': self.hparams.patience // 2,
            'factor': 0.5,
        }
        backward_scheduler = {
            'scheduler': ReduceLROnPlateau(backward_opt, **lr_args),
            'name': 'lr_scheduler_backward',
        }
        forward_scheduler = {
            'scheduler': ReduceLROnPlateau(forward_opt, **lr_args),
            'name': 'lr_scheduler_forward',
        }
        
        return [backward_opt, forward_opt], [backward_scheduler, forward_scheduler]
        
class ClassicSchrodingerBridge(StandardSchrodingerBridge):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @torch.no_grad()
    def go_forward(self, xk : Tensor, k : int) -> Tensor:
        ks = self.k_to_tensor(k, xk.size(0))
        xk_plus_one = self.forward_model(xk, ks) + torch.sqrt(2 * self.gammas[k + 1]) * torch.randn_like(xk)
        
        return xk_plus_one
    
    @torch.no_grad()
    def go_backward(self, xk_plus_one : Tensor, k_plus_one : int) -> Tensor:
        ks_plus_one = self.k_to_tensor(k_plus_one, xk_plus_one.size(0))
        xk = self.backward_model(xk_plus_one, ks_plus_one) + torch.sqrt(2 * self.gammas[k_plus_one]) * torch.randn_like(xk_plus_one)
        
        return xk
    
    def _forward_loss(self, xk_plus_one : Tensor, k_plus_one : int, xN : Tensor) -> Tuple[Tensor, Tensor]:
        xk = self.go_backward(xk_plus_one, k_plus_one)
        ks = self.k_to_tensor(k_plus_one - 1, xk_plus_one.size(0))
        xk_plus_one_prime = self.forward_model(xk, ks)
        loss = self.mse(xk_plus_one_prime, xk_plus_one)
        
        return loss, xk
    
    def _backward_loss(self, xk : Tensor, k : int, x0 : Tensor) -> Tuple[Tensor, Tensor]:
        xk_plus_one = self.go_forward(xk, k)
        ks_plus_one = self.k_to_tensor(k, xk.size(0))
        xk_prime = self.backward_model(xk_plus_one, ks_plus_one)
        loss = self.mse(xk_prime, xk)
        
        return loss, xk_plus_one
    
class BaseSimplifiedSchrodingerBridge(StandardSchrodingerBridge):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gammas_bar = torch.cumsum(self.gammas, 0)
        self.sigma_backward = 2 * self.gammas[1:] * self.gammas_bar[:-1] / self.gammas_bar[1:]
        self.sigma_forward = 2 * self.gammas[1:] * (1 - self.gammas_bar[1:]) / (1 - self.gammas_bar[:-1])
    
class TRSchrodingerBridge(BaseSimplifiedSchrodingerBridge):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @torch.no_grad()
    def go_forward(self, xk : Tensor, k : int) -> Tensor:
        batch_size = xk.size(0)
        ks = self.k_to_tensor(k, batch_size)
        xN = self.forward_model(xk, ks)
        mu = xk + self.gammas[k + 1] / (1 - self.gammas_bar[k]) * (xN - xk)
        sigma = self.sigma_forward[k]
        xk_plus_one = mu + sigma * torch.randn_like(xk)
        
        return xk_plus_one
    
    @torch.no_grad()
    def go_backward(self, xk_plus_one : Tensor, k_plus_one : int) -> Tensor:
        batch_size = xk_plus_one.size(0)
        ks_plus_one = self.k_to_tensor(k_plus_one, batch_size)
        x0 = self.backward_model(xk_plus_one, ks_plus_one)
        mu = xk_plus_one + self.gammas[k_plus_one] / self.gammas_bar[k_plus_one] * (x0 - xk_plus_one)
        sigma = self.sigma_backward[k_plus_one - 1]
        xk = mu + sigma * torch.randn_like(xk_plus_one)
        
        return xk
    
    def _backward_loss(self, xk : Tensor, k : int, x0 : Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = xk.size(0)
        xk_plus_one = self.go_forward(xk, k)
        ks_plus_one = self.k_to_tensor(k + 1, batch_size)
        x0_pred = self.backward_model(xk_plus_one, ks_plus_one)
        loss = self.mse(x0_pred, x0)
        
        return loss, xk_plus_one
    
    def _forward_loss(self, xk_plus_one : Tensor, k_plus_one : int, xN : Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = xk_plus_one.size(0)
        xk = self.go_backward(xk_plus_one, k_plus_one)
        ks = self.k_to_tensor(k_plus_one - 1, batch_size)
        xN_pred = self.forward_model(xk, ks)
        loss = self.mse(xN_pred, xN)
        
        return loss, xk
        
class FRSchrodingerBridge(BaseSimplifiedSchrodingerBridge):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @torch.no_grad()
    def go_forward(self, xk : Tensor, k : int) -> Tensor:
        batch_size = xk.size(0)
        ks = self.k_to_tensor(k, batch_size)
        f = self.forward_model(xk, ks)
        mu = xk + self.gammas[k + 1] * f
        sigma = self.sigma_forward[k]
        xk_plus_one = mu + sigma * torch.randn_like(xk)
        
        return xk_plus_one
    
    @torch.no_grad()
    def go_backward(self, xk_plus_one : Tensor, k_plus_one : int) -> Tensor:
        batch_size = xk_plus_one.size(0)
        ks_plus_one = self.k_to_tensor(k_plus_one, batch_size)
        b = self.backward_model(xk_plus_one, ks_plus_one)
        mu = xk_plus_one + self.gammas[k_plus_one] * b
        sigma = self.sigma_backward[k_plus_one - 1]
        xk = mu + sigma * torch.randn_like(xk_plus_one)
        
        return xk
    
    def _backward_loss(self, xk : Tensor, k : int, x0 : Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = xk.size(0)
        xk_plus_one = self.go_forward(xk, k)
        target = (x0 - xk_plus_one) / self.gammas_bar[k + 1]
        ks_plus_one = self.k_to_tensor(k + 1, batch_size)
        pred = self.backward_model(xk_plus_one, ks_plus_one)
        loss = self.mse(pred, target)
        
        return loss, xk_plus_one
    
    def _forward_loss(self, xk_plus_one : Tensor, k_plus_one : int, xN : Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = xk_plus_one.size(0)
        xk = self.go_backward(xk_plus_one, k_plus_one)
        target = (xN - xk) / (1 - self.gammas_bar[k_plus_one - 1])
        ks = self.k_to_tensor(k_plus_one - 1, batch_size)
        pred = self.forward_model(xk, ks)
        loss = self.mse(pred, target)
        
        return loss, xk