from diffusers import UNet2DModel
import torch

class Unet2D(UNet2DModel):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def forward(self, x : torch.Tensor, time_step : torch.Tensor):
        time_step = time_step.flatten()
        return super().forward(x, time_step).sample