import pytorch_lightning as pl
import torch
from typing import Tuple, Any, Callable
from torch import Tensor
from src.lightning_modules.baselightningmodule import BaseLightningModule

class StandardSchrodingerBridge(BaseLightningModule):
    def __init__(
        self,
        forward_model : torch.nn.Module,
        backward_model : torch.nn.Module,
        max_gamma : float = None,
        min_gamma : float = None,
        num_steps : int = None,
        training_backward : bool = True,
        lr : float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore = ["forward_model", "backward_model"])
        self.automatic_optimization : bool = False
        
        min_gamma, max_gamma, num_steps = self._check_gammas(min_gamma, max_gamma, num_steps)
        self.hparams.update({
            "min_gamma" : min_gamma,
            "max_gamma" : max_gamma,
            "num_steps" : num_steps,
        })
        
        calculate_gammas = lambda x : - abs((max_gamma - min_gamma) * (2 * x - num_steps) / num_steps) + max_gamma
        self.gammas = calculate_gammas(torch.arange(num_steps))
        assert (torch.sum(self.gammas) - 1).abs() < 1e-3, "Gammas must sum to 1"

        self.gammas = torch.cat([
            torch.tensor([0]),
            torch.linspace(min_gamma, max_gamma, num_steps // 2), 
            torch.linspace(max_gamma, min_gamma, num_steps // 2)
            ])
        
        self.forward_model : Callable[[Tensor, Tensor], Tensor] = forward_model
        self.backward_model : Callable[[Tensor, Tensor], Tensor] = backward_model
        
        self.mse : Callable[[Tensor, Tensor], Tensor] = torch.nn.MSELoss()        
        self.losses : list = []
        
    def _check_gammas(self, min_gamma, max_gamma, num_steps):
        assert sum([val is None for val in [num_steps, min_gamma, max_gamma]]) < 2, "There must be less than 2 None values"
        
        # min_gamma + max_gamma  = 2 / num_steps
        
        if num_steps is None:
            num_steps = 2 / (min_gamma + max_gamma)
        elif min_gamma is None:
            min_gamma = 2 / num_steps - max_gamma
        elif max_gamma is None:
            max_gamma = 2 / num_steps - min_gamma
        
        num_steps = int(num_steps)
        
        assert max_gamma >= min_gamma, "max_gamma must be greater than min_gamma"
        assert min_gamma > 0, "min_gamma must be greater than 0"
        
        return min_gamma, max_gamma, num_steps
        
    def k_to_tensor(self, k : int, size : Tuple[int]) -> Tensor:
        return torch.full((size, 1), k, dtype = torch.float32, device = self.device)
    
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
            
        if return_trajectory:
            return trajectory
                
        return xk
    
    def training_step(self, batch : Tensor, batch_idx : int) -> None:
        backward_opt, forward_opt = self.optimizers()
        epoch_losses = torch.zeros(self.hparams.num_steps)

        if self.hparams.training_backward:
            x0 = batch
            xk = x0.clone()
            range_ = range(self.hparams.num_steps) # 0, 1, 2, ..., num_steps - 1
        else:
            xN = batch
            xk_plus_one = xN.clone()
            range_ = reversed(range(self.hparams.num_steps))  # num_steps - 1, num_steps - 2, ..., 1, 0
        
        for i, k in enumerate(range_):
            if self.hparams.training_backward:
                # training the backward_model                
                loss, xk_plus_one = self._backward_loss(xk, k, x0)
                
                backward_opt.zero_grad()
                self.manual_backward(loss)
                backward_opt.step()
                
                xk = xk_plus_one
            
            else:
                # training the forward_model
                loss, xk = self._forward_loss(xk_plus_one, k + 1, xN)
                
                forward_opt.zero_grad()
                self.manual_backward(loss)
                forward_opt.step()
                
                xk_plus_one = xk
                
            epoch_losses[i] = loss.item()
            
        avg_loss = epoch_losses.mean().item()
        self.losses.append(avg_loss)      
        
        self.log_dict({
            "training_backward" : self.hparams.training_backward,
            "Average backward loss" : avg_loss * self.hparams.training_backward,
            "Average forward loss" : avg_loss * (not self.hparams.training_backward),
            **{f"backward_loss_step/{k}" : l * self.hparams.training_backward for k, l in enumerate(epoch_losses)},
            **{f"forward_loss_step/{k}" : l * (not self.hparams.training_backward) for k, l in enumerate(epoch_losses)},
        }, prog_bar = True)
                                             
    def configure_optimizers(self):
        backward_opt = torch.optim.Adam(self.backward_model.parameters(), lr = self.hparams.lr)
        forward_opt = torch.optim.Adam(self.forward_model.parameters(), lr = self.hparams.lr)
        
        return [backward_opt, forward_opt], []
        
class ClassicSchrodingerBridge(StandardSchrodingerBridge):
    def __init__(**kwargs):
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
    
    
if __name__ == "__main__":
    min_gamma = 1e-3