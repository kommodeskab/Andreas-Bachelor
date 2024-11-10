from src.lightning_modules import TRDSB, DDPM
from torch import Tensor
import torch
import os
    

class PretrainedTRDSB(TRDSB):
    def __init__(self, ddpm_model : str, **kwargs):
        super().__init__(**kwargs, initial_forward_sampling = None)
        
        model_dict = {
            "cat_32": "091124235944"
        }
        
        folder_name = f"logs/diffusion/{model_dict[ddpm_model]}/checkpoints"
        ckpt_path = os.path.join(folder_name, os.listdir(folder_name)[0])
        
        self.ddpm = DDPM.load_from_checkpoint(
            checkpoint_path = ckpt_path,
            model = self.forward_model,
            optimizer = None,
            strict=True,
            )
        self.ddpm.set_timesteps(self.hparams.num_steps)
    
    @torch.no_grad()
    def sample(self, x_start : Tensor, forward : bool = True, return_trajectory : bool = False, clamp : bool = False, ema_scope : bool = False) -> Tensor:
        # in the very first iteration, use the pretrained DDPM model to sample
        if self.hparams.DSB_iteration == 1 and self.hparams.training_backward and forward:
            return self.ddpm.sample(x_start, return_trajectory = return_trajectory, clamp = clamp)
        return super().sample(x_start, forward = forward, return_trajectory = return_trajectory, clamp = clamp, ema_scope = ema_scope)