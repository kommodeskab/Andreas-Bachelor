from src.lightning_modules import TRDSB, DDPM
from torch import Tensor
import torch
from src.lightning_modules.utils import ckpt_path_from_id

class PPTRDSB(TRDSB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, initial_forward_sampling = None)
    
    def _get_pseudo_xN(self, x0 : Tensor) -> Tensor:
        """
        Given some batched x0, return a "pseudo" terminal point
        For example, add noise, or find the closest point in the end dataset, etc.
        """
        raise NotImplementedError
    
    def forward_call(self, xk : Tensor, k : Tensor) -> Tensor:        
        if self.hparams.training_backward and self.hparams.DSB_iteration == 1:
            if k[0].item() == 0:
                self.pseudo_xN = self._get_pseudo_xN(xk)
                
            return self.pseudo_xN
        
        return super().forward_call(xk, k)

class DDPMPretrainedTRDSB(TRDSB):
    def __init__(self, forward_model_id : str, **kwargs):
        super().__init__(**kwargs)
        
        ckpt_path = ckpt_path_from_id(forward_model_id)
        
        self.ddpm = DDPM.load_from_checkpoint(
            checkpoint_path = ckpt_path,
            model = self.forward_model,
            optimizer = None,
            strict=True,
            )
        self.ddpm.set_timesteps(self.hparams.num_steps)

    @torch.no_grad()
    def sample(self, x_start : Tensor, forward : bool = True, return_trajectory : bool = False, clamp : bool = False, ema_scope : bool = False) -> Tensor:
        if (
            self.hparams.DSB_iteration == 1 and
            self.hparams.training_backward and
            forward
        ):
            return self.ddpm.sample(x_start, return_trajectory, clamp)
    
        return super().sample(x_start, forward, return_trajectory, clamp, ema_scope)

class DDPMPPTRDSB(PPTRDSB):
    def __init__(self, forward_model_id : str, **kwargs):
        super().__init__(**kwargs)
        
        ckpt_path = ckpt_path_from_id(forward_model_id)
        
        self.ddpm = DDPM.load_from_checkpoint(
            checkpoint_path = ckpt_path,
            model = self.forward_model,
            optimizer = None,
            strict=True,
            )
        self.ddpm.set_timesteps(self.hparams.num_steps)
        
    def _get_pseudo_xN(self, x0 : Tensor) -> Tensor:
        return self.ddpm.sample(x0, return_trajectory=False, clamp=True)