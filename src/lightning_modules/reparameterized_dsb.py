from src.lightning_modules.schrodinger_bridge import StandardDSB
from torch import Tensor
import torch
from typing import Tuple

class BaseReparameterizedDSB(StandardDSB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gammas_bar = torch.cumsum(self.gammas, 0)
        self.sigma_backward = 2 * self.gammas[1:] * self.gammas_bar[:-1] / self.gammas_bar[1:]
        self.sigma_forward = 2 * self.gammas[1:] * (1 - self.gammas_bar[1:]) / (1 - self.gammas_bar[:-1])

class TRDSB(BaseReparameterizedDSB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def go_forward(self, xk : Tensor, k : int) -> Tensor:
        if self.DSB_iteration == 0 and self.hparams.training_backward:
            return self.ornstein_uhlenbeck(xk, k)
        
        batch_size = xk.size(0)
        ks = self.k_to_tensor(k, batch_size)
        xN_pred = self.forward_call(xk, ks)
        mu = xk + self.gammas[k + 1] / (1 - self.gammas_bar[k]) * (xN_pred - xk)
        sigma = self.sigma_forward[k]
        xk_plus_one = mu + sigma * torch.randn_like(xk)
        
        return xk_plus_one
    
    def go_backward(self, xk_plus_one : Tensor, k_plus_one : int) -> Tensor:
        batch_size = xk_plus_one.size(0)
        ks_plus_one = self.k_to_tensor(k_plus_one, batch_size)
        x0_pred = self.backward_call(xk_plus_one, ks_plus_one)
        mu = xk_plus_one + self.gammas[k_plus_one] / self.gammas_bar[k_plus_one] * (x0_pred - xk_plus_one)
        sigma = self.sigma_backward[k_plus_one - 1]
        xk = mu + sigma * torch.randn_like(xk_plus_one)
        
        return xk
    
    def _forward_loss(self, xk : Tensor, ks : Tensor, xN : Tensor) -> Tensor:
        xN_pred = self.forward_call(xk, ks)
        loss = self.mse(xN_pred, xN)
        return loss
    
    def _backward_loss(self, xk_plus_one : Tensor, ks_plus_one : Tensor, x0 : Tensor) -> Tensor:
        x0_pred = self.backward_call(xk_plus_one, ks_plus_one)
        loss = self.mse(x0_pred, x0)
        return loss
    
class FRDSB(BaseReparameterizedDSB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def go_forward(self, xk : Tensor, k : int) -> Tensor:
        if self.DSB_iteration == 0 and self.hparams.training_backward:
            return self.ornstein_uhlenbeck(xk, k)
        
        batch_size = xk.size(0)
        ks = self.k_to_tensor(k, batch_size)
        f = self.forward_call(xk, ks)
        mu = xk + self.gammas[k + 1] * f
        sigma = self.sigma_forward[k]
        xk_plus_one = mu + sigma * torch.randn_like(xk)
        
        return xk_plus_one
    
    def go_backward(self, xk_plus_one : Tensor, k_plus_one : int) -> Tensor:
        batch_size = xk_plus_one.size(0)
        ks_plus_one = self.k_to_tensor(k_plus_one, batch_size)
        b = self.backward_call(xk_plus_one, ks_plus_one)
        mu = xk_plus_one + self.gammas[k_plus_one] * b
        sigma = self.sigma_backward[k_plus_one - 1]
        xk = mu + sigma * torch.randn_like(xk_plus_one)
        
        return xk
    
    def _forward_loss(self, xk : Tensor, ks : Tensor, xN : Tensor) -> Tensor:
        target = (xN - xk) / (1 - self.gammas_bar[ks])
        pred = self.forward_call(xk, ks)
        loss = self.mse(target, pred)
        return loss
    
    def _backward_loss(self, xk_plus_one : Tensor, ks_plus_one : Tensor, x0 : Tensor) -> Tensor:
        target = (x0 - xk_plus_one) / self.gammas_bar[ks_plus_one]
        pred = self.backward_call(xk_plus_one, ks_plus_one)
        loss = self.mse(target, pred)
        return loss