import torch
from typing import Tuple, Any, Callable
from torch import Tensor
from src.lightning_modules.baselightningmodule import BaseLightningModule
from torch.optim import Adam, AdamW
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.nn import Module

class StandardDSB(BaseLightningModule):
    def __init__(
        self,
        forward_model : torch.nn.Module,
        backward_model : torch.nn.Module,
        max_gamma : float,
        min_gamma : float,
        num_steps : int,
        patience: int,
        max_evals: int | None = None,
        strict_gammas : bool = True,
        lr : float = 1e-3,
        max_norm : float = 0.5,
    ):
        """
        Initializes the StandardDSB model

        Args:
            forward_model (torch.nn.Module): the forward model
            backward_model (torch.nn.Module): the backward model
            max_gamma (float): the maximum gamma
            min_gamma (float): the minimum gamma
            num_steps (int): the number of steps
            patience (int): the patience
            max_batches (int): the maximum number of batches
            strict_gammas (bool): whether to enforce the sum of gammas to be equal to 1
            lr (float): the learning rate
            max_norm (float): the maximum norm for clipping the gradients
        """
        super().__init__()
        self.save_hyperparameters(ignore = ["forward_model", "backward_model"])
        self.automatic_optimization : bool = False
        self.hparams["training_backward"] = True

        assert max_gamma >= min_gamma, f"{max_gamma = } must be greater than {min_gamma = }"
        first_steps = num_steps // 2
        gammas = torch.zeros(num_steps + 1)
        gammas[1:first_steps + 1] = torch.linspace(min_gamma, max_gamma, first_steps)
        gammas[first_steps + 1:] = torch.linspace(max_gamma, min_gamma, num_steps - first_steps)
        self.gammas = gammas
        
        if strict_gammas:
            gammas_sum = gammas.sum().item()
            assert abs(1 - gammas_sum) < 1e-3, f"The sum of gammas must be equal to 1, but got {gammas_sum = }"
        
        self.forward_model : Callable[[Tensor, Tensor], Tensor] = forward_model
        self.backward_model : Callable[[Tensor, Tensor], Tensor] = backward_model
        
        self.mse : Callable[[Tensor, Tensor], Tensor] = torch.nn.MSELoss()        
        self.val_losses : list = []
        self.DSB_iteration : int = 0

    def on_train_start(self) -> None:
        assert self.trainer.datamodule.hparams.training_backward == self.hparams.training_backward, "The training direction must be the same for datamodule and model"

    def has_converged(self) -> bool:
        """
        Determines if the model has converged based on the validation losses.
        Can be overriden to implement custom convergence criteria.
    
        Checks if the validation losses have increased for 'patience' epochs.
        If not, the model has converged.
        """
        losses, patience, max_evals = self.val_losses, self.hparams.patience, self.hparams.max_evals

        if max_evals is not None:
            if len(losses) >= max_evals:
                return True

        if len(losses) < patience + 1:
            return False
        
        min_loss = min(losses[:-patience])
        return all([l > min_loss for l in losses[-patience:]])
    
    def on_train_batch_start(self, batch : Tensor, batch_idx : int) -> None:
        """
        We check if the model has converged before each batch
        if the model has converged we do the following:
        - switch the training direction
        - reset the val_losses
        - reset the learning rate of the optimizer
        - reset the learning rate scheduler
        
        we also return -1 to skip the rest of the epoch if converged (pytorch lightning behavior)
        
        """

        if self.has_converged():
            if not self.hparams.training_backward: 
                self.DSB_iteration += 1
                
            self.hparams.training_backward = not self.hparams.training_backward
            self.trainer.datamodule.hparams.training_backward = self.hparams.training_backward
            self.val_losses = []
            
            # resetting the learning rate
            for optimizer in self.optimizers():
                for pg in optimizer.param_groups:
                    pg['lr'] = self.hparams.lr
                    
            # resetting the learning rate scheduler
            # only works for ReduceLROnPlateau
            for scheduler in self.lr_schedulers():
                scheduler.num_bad_epochs = 0
            
            return -1
    
    def k_to_tensor(self, k : int, size : Tuple[int]) -> Tensor:
        """
        Given k, return a tensor of size 'size' filled with k

        :param int k: the value to fill the tensor with
        :param Tuple[int] size: the size of the tensor

        :return Tensor: the tensor filled with k
        """
        return torch.full((size, ), k, dtype = torch.float32, device = self.device)
    
    def forward_call(self, x : Tensor, k: Tensor) -> Tensor:
        """
        Calls the forward model
        """
        return self.forward_model(x, k)
    
    def backward_call(self, x : Tensor, k : Tensor) -> Tensor:
        """
        Calls the backward model
        """
        return self.backward_model(x, k)
    
    def ornstein_uhlenbeck(self, xk : Tensor, k : int) -> Tensor:
        alpha = 0.1
        mu = xk - self.gammas[k + 1] * alpha * xk
        sigma = torch.sqrt(2 * self.gammas[k + 1])
        return mu + sigma * torch.randn_like(xk)
    
    def brownian(self, xk : Tensor, k : int) -> Tensor:
        sigma = torch.sqrt(2 * self.gammas[k + 1])
        return xk + sigma * torch.randn_like(xk)
        
    def go_forward(self, xk : Tensor, k : int) -> Tensor:
        """        
        Get :math:`x_{k + 1} | x_{k}`
        
        :param Tensor xk: the current point
        :param int k: the current step
        
        :return xk_plus_one: the next point
        """
        raise NotImplementedError
    
    def go_backward(self, xk_plus_one : Tensor, k_plus_one : int) -> Tensor:
        """
        Get :math:`x_{k} | x_{k + 1}`
        
        :param Tensor xk_plus_one: the current point
        :param int k_plus_one: the current step
        
        :return xk: the previous point
        """
        raise NotImplementedError
    
    def _forward_loss(self, xk_plus_one : Tensor, ks_plus_one : int, xN : Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute the loss for the forward model.
        Uses the backward model to 'walk' backwards and optimize the forward model to 'walk' back to the original point
        
        :param Tensor xk_plus_one: the current point
        :param Tensor ks_plus_one: the current step
        :param Tensor xN: the end point
        
        :return loss and xk: the loss and the previous point
        """
        raise NotImplementedError
    
    def _backward_loss(self, xk : Tensor, ks : int, x0 : Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute the loss for the backward model.
        Uses the forward model to 'walk' forward and optimize the backward model to 'walk' back to the original point
        
        :param Tensor xk: the current point
        :param Tensor ks: the current step
        :param Tensor x0: the start point
        
        :return loss and xk_plus_one: the loss and the next point
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def sample(self, x_start : Tensor, forward : bool = True, return_trajectory : bool = False) -> Tensor:
        """
        Given the start point x_start, sample the final point xN / x0 by going forward / backward
        Also, return the trajectory if return_trajectory is True
        
        :param Tensor x_start: the start point
        :param bool forward: whether to go forward or backward
        :param bool return_trajectory: whether to return the trajectory
        
        :return Tensor: the final point xN / x0 or the trajectory
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
    
    def _optimize(self, loss : Tensor, optimizer : Optimizer, model : Module) -> None:
        """
        Given the loss, optimizer and model, optimize the model
        
        :param Tensor loss: the loss
        :param Optimizer optimizer: the optimizer
        :param Module model: the model
        """
        optimizer.zero_grad()
        self.manual_backward(loss)
        clip_grad_norm_(model.parameters(), self.hparams.max_norm)
        optimizer.step()
    
    def training_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        backward_opt, forward_opt = self.optimizers()
        
        if not isinstance(batch, int):
            self.cache = self.sample(batch, forward = self.hparams.training_backward, return_trajectory = True)
        
        trajectory = self.cache
            
        x0, xN = trajectory[0], trajectory[-1]
        traj_len, batch_size = trajectory.size(0), trajectory.size(1)
        
        ks = torch.randint(0, traj_len - 1, (batch_size,))
        if self.hparams.training_backward:
            ks += 1
            
        sampled_batch = trajectory[ks, torch.arange(batch_size)].requires_grad_()
        
        if self.hparams.training_backward:
            loss = self._backward_loss(sampled_batch, ks, x0)
            self._optimize(loss, backward_opt, self.backward_model)
            self.log(f"backward_loss_{self.DSB_iteration}/train", loss, on_step = True, on_epoch = False)
        else:
            loss = self._forward_loss(sampled_batch, ks, xN)
            self._optimize(loss, forward_opt, self.forward_model)
            self.log(f"forward_loss_{self.DSB_iteration}/train", loss, on_step = True, on_epoch = False)
                
    def validation_step(self, batch : Tensor, batch_idx : int, dataloader_idx : int) -> Tensor:
        trajectory = self.sample(batch, forward = self.hparams.training_backward, return_trajectory = True)
        batch_size = trajectory.size(1)
        x0, xN = trajectory[0], trajectory[-1]
        
        if self.hparams.training_backward and dataloader_idx == 0:
            for k in range(1, self.hparams.num_steps + 1):
                ks = self.k_to_tensor(k, batch_size)
                loss = self._backward_loss(trajectory[k], ks, x0)
                self.log(f"backward_loss_{self.DSB_iteration}/val", loss, prog_bar=True, add_dataloader_idx=False)
        elif not self.hparams.training_backward and dataloader_idx == 1:
            for k in range(0, self.hparams.num_steps):
                ks = self.k_to_tensor(k, batch_size)
                loss = self._forward_loss(trajectory[k], ks, xN)
                self.log(f"forward_loss_{self.DSB_iteration}/val", loss, prog_bar=True, add_dataloader_idx=False)
 
    def on_validation_epoch_end(self) -> None:
        # after validation we want to update the learning rate
        backward_scheduler, forward_scheduler = self.lr_schedulers()
        metrics = self.trainer.callback_metrics

        # if no losses were computed, return
        if len(metrics) == 0:
            return
        
        # take a step based on the validation loss
        if self.hparams.training_backward:
            val_loss = metrics[f"backward_loss_{self.DSB_iteration}/val"].item()
            backward_scheduler.step(val_loss)
        else:
            val_loss = metrics[f"forward_loss_{self.DSB_iteration}/val"].item()
            forward_scheduler.step(val_loss)

        self.val_losses.append(val_loss)
        
    def configure_optimizers(self):
        backward_opt = AdamW(self.backward_model.parameters(), lr = self.hparams.lr)
        forward_opt = AdamW(self.forward_model.parameters(), lr = self.hparams.lr)
        
        # convergence will happen after 'patience' validation steps with no improvement
        # therefore reduce the lr after 'patience // 2' validation steps with no improvement
        lr_args = {'patience': self.hparams.patience // 2, 'factor': 0.5}
        backward_scheduler = {'scheduler': ReduceLROnPlateau(backward_opt, **lr_args), 'name': 'lr_scheduler_backward'}
        forward_scheduler = {'scheduler': ReduceLROnPlateau(forward_opt, **lr_args), 'name': 'lr_scheduler_forward'}
        
        return [backward_opt, forward_opt], [backward_scheduler, forward_scheduler]