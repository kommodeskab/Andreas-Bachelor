from src.lightning_modules.schrodinger_bridge import StandardDSB
from torch import Tensor
import torch

class BaseReparameterizedDSB(StandardDSB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gammas_bar = torch.cumsum(self.gammas, 0)
        self.sigma_backward = 2 * self.gammas[1:] * self.gammas_bar[:-1] / self.gammas_bar[1:]
        self.sigma_forward = 2 * self.gammas[1:] * (1 - self.gammas_bar[1:]) / (1 - self.gammas_bar[:-1])
        self.sigma_backward = torch.cat([torch.tensor([0.0]).to(self.device), self.sigma_backward])
        self.sigma_forward = torch.cat([torch.tensor([0.0]).to(self.device), self.sigma_forward])

class TRDSB(BaseReparameterizedDSB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def go_forward(self, xk : Tensor, k : int) -> Tensor:
        batch_size = xk.size(0)
        ks = self.k_to_tensor(k, batch_size)
        xN_pred = self.forward_call(xk, ks)
        mu = xk + self.gammas[k + 1] / (1 - self.gammas_bar[k]) * (xN_pred - xk)
        sigma = self.sigma_forward[k + 1]
        xk_plus_one = mu + sigma * torch.randn_like(xk)
        
        return xk_plus_one
    
    def go_backward(self, xk_plus_one : Tensor, k_plus_one : int) -> Tensor:
        batch_size = xk_plus_one.size(0)
        ks_plus_one = self.k_to_tensor(k_plus_one, batch_size)
        x0_pred = self.backward_call(xk_plus_one, ks_plus_one)
        mu = xk_plus_one + self.gammas[k_plus_one] / self.gammas_bar[k_plus_one] * (x0_pred - xk_plus_one)
        sigma = self.sigma_backward[k_plus_one]
        xk = mu + sigma * torch.randn_like(xk_plus_one)
        
        return xk
    
    def _forward_loss(self, xk : Tensor, ks : Tensor, xN : Tensor) -> Tensor:
        xN_pred = self.forward_call(xk, ks)
        loss = self.mse(xN_pred, xN)
        return loss
    
    def _backward_loss(self, xk : Tensor, ks : Tensor, x0 : Tensor) -> Tensor:
        x0_pred = self.backward_call(xk, ks)
        loss = self.mse(x0_pred, x0)
        return loss
    
    def pred_x0(self, xk : Tensor, ks : Tensor) -> Tensor:
        return self.backward_call(xk, ks)
    
    def pred_xN(self, xk : Tensor, ks : Tensor) -> Tensor:
        return self.forward_call(xk, ks)
    
class FRDSB(BaseReparameterizedDSB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def go_forward(self, xk : Tensor, k : int) -> Tensor:
        batch_size = xk.size(0)
        ks = self.k_to_tensor(k, batch_size)
        f = self.forward_call(xk, ks)
        mu = xk + self.gammas[k + 1] * f
        sigma = self.sigma_forward[k + 1]
        xk_plus_one = mu + sigma * torch.randn_like(xk)
        
        return xk_plus_one
    
    def go_backward(self, xk_plus_one : Tensor, k_plus_one : int) -> Tensor:
        batch_size = xk_plus_one.size(0)
        ks_plus_one = self.k_to_tensor(k_plus_one, batch_size)
        b = self.backward_call(xk_plus_one, ks_plus_one)
        mu = xk_plus_one + self.gammas[k_plus_one] * b
        sigma = self.sigma_backward[k_plus_one]
        xk = mu + sigma * torch.randn_like(xk_plus_one)
        
        return xk
    
    def _forward_loss(self, xk : Tensor, ks : Tensor, xN : Tensor) -> Tensor:
        shape_for_constant = self._get_shape_for_constant(xk)
        gammas_bar = self.gammas_bar.to(self.device)[ks].view(shape_for_constant)
        target = (xN - xk) / (1 - gammas_bar)
        pred = self.forward_call(xk, ks)
        loss = self.mse(target, pred)
        return loss
    
    def _backward_loss(self, xk : Tensor, ks : Tensor, x0 : Tensor) -> Tensor:
        shape_for_constant = self._get_shape_for_constant(xk)
        gammas_bar = self.gammas_bar.to(self.device)[ks].view(shape_for_constant)
        target = (x0 - xk) / gammas_bar
        pred = self.backward_call(xk, ks)
        loss = self.mse(target, pred)
        return loss
    
    def pred_x0(self, xk : Tensor, ks : Tensor) -> Tensor:
        shape_for_constant = self._get_shape_for_constant(xk)
        gammas_bar = self.gammas_bar.to(self.device)[ks].view(shape_for_constant)
        return xk - gammas_bar * self.backward_call(xk, ks)
    
    def pred_xN(self, xk : Tensor, ks : Tensor) -> Tensor:
        shape_for_constant = self._get_shape_for_constant(xk)
        gammas_bar = self.gammas_bar.to(self.device)[ks].view(shape_for_constant)
        return xk + (1 - gammas_bar) * self.forward_call(xk, ks)
    
    def _get_shape_for_constant(self, x : Tensor) -> list[int]:
        return [-1] + [1] * (x.dim() - 1)