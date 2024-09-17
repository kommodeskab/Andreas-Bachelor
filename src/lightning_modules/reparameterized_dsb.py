from src.lightning_modules.schrodinger_bridge import BaseReparameterizedDSB
from torch import Tensor
import torch
from typing import Tuple

class TRDSB(BaseReparameterizedDSB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @torch.no_grad()
    def go_forward(self, xk : Tensor, k : int) -> Tensor:
        batch_size = xk.size(0)
        ks = self.k_to_tensor(k, batch_size)
        xN_pred = self.forward_call(xk, ks)
        mu = xk + self.gammas[k + 1] / (1 - self.gammas_bar[k]) * (xN_pred - xk)
        sigma = self.sigma_forward[k]
        xk_plus_one = mu + sigma * torch.randn_like(xk)
        
        return xk_plus_one
    
    @torch.no_grad()
    def go_backward(self, xk_plus_one : Tensor, k_plus_one : int) -> Tensor:
        batch_size = xk_plus_one.size(0)
        ks_plus_one = self.k_to_tensor(k_plus_one, batch_size)
        x0_pred = self.backward_call(xk_plus_one, ks_plus_one)
        mu = xk_plus_one + self.gammas[k_plus_one] / self.gammas_bar[k_plus_one] * (x0_pred - xk_plus_one)
        sigma = self.sigma_backward[k_plus_one - 1]
        xk = mu + sigma * torch.randn_like(xk_plus_one)
        
        return xk
    
    def _backward_loss(self, xk : Tensor, k : int, x0 : Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = xk.size(0)
        xk_plus_one = self.go_forward(xk, k)
        ks_plus_one = self.k_to_tensor(k + 1, batch_size)
        x0_pred = self.backward_call(xk_plus_one, ks_plus_one)
        loss = self.mse(x0_pred, x0)
        
        return loss, xk_plus_one
    
    def _forward_loss(self, xk_plus_one : Tensor, k_plus_one : int, xN : Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = xk_plus_one.size(0)
        xk = self.go_backward(xk_plus_one, k_plus_one)
        ks = self.k_to_tensor(k_plus_one - 1, batch_size)
        xN_pred = self.forward_call(xk, ks)
        loss = self.mse(xN_pred, xN)
        
        return loss, xk
    
class TRDSBWithHack(TRDSB):
    def __init__(self, **kwargs):
        self.hack_permitted : bool = True
        super().__init__(**kwargs)
    
    def sample(self, x_start: Tensor, forward: bool = True, return_trajectory: bool = False) -> Tensor:
        self.hack_permitted = False
        return super().sample(x_start, forward, return_trajectory)
    
    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        self.last_batch = batch
        self.hack_permitted = True
        return super().training_step(batch, batch_idx)
    
    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        self.last_batch = batch
        self.hack_permitted = True
        return super().validation_step(batch, batch_idx)
    
    def forward_call(self, x: Tensor, k: Tensor):
        if self.training_backward and self.DSB_iteration == 0 and self.hack_permitted:
            return self.last_batch["xN"]
        
        return super().forward_call(x, k)
    
class FRDSB(BaseReparameterizedDSB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @torch.no_grad()
    def go_forward(self, xk : Tensor, k : int) -> Tensor:
        batch_size = xk.size(0)
        ks = self.k_to_tensor(k, batch_size)
        f = self.forward_call(xk, ks)
        mu = xk + self.gammas[k + 1] * f
        sigma = self.sigma_forward[k]
        xk_plus_one = mu + sigma * torch.randn_like(xk)
        
        return xk_plus_one
    
    @torch.no_grad()
    def go_backward(self, xk_plus_one : Tensor, k_plus_one : int) -> Tensor:
        batch_size = xk_plus_one.size(0)
        ks_plus_one = self.k_to_tensor(k_plus_one, batch_size)
        b = self.backward_call(xk_plus_one, ks_plus_one)
        mu = xk_plus_one + self.gammas[k_plus_one] * b
        sigma = self.sigma_backward[k_plus_one - 1]
        xk = mu + sigma * torch.randn_like(xk_plus_one)
        
        return xk
    
    def _backward_loss(self, xk : Tensor, k : int, x0 : Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = xk.size(0)
        xk_plus_one = self.go_forward(xk, k)
        target = (x0 - xk_plus_one) / self.gammas_bar[k + 1]
        ks_plus_one = self.k_to_tensor(k + 1, batch_size)
        pred = self.backward_call(xk_plus_one, ks_plus_one)
        loss = self.mse(pred, target)
        
        return loss, xk_plus_one
    
    def _forward_loss(self, xk_plus_one : Tensor, k_plus_one : int, xN : Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = xk_plus_one.size(0)
        xk = self.go_backward(xk_plus_one, k_plus_one)
        target = (xN - xk) / (1 - self.gammas_bar[k_plus_one - 1])
        ks = self.k_to_tensor(k_plus_one - 1, batch_size)
        pred = self.forward_call(xk, ks)
        loss = self.mse(pred, target)
        
        return loss, xk