from diffusers import UNet2DModel, UNet1DModel
import torch

class UNet2D(UNet2DModel):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def forward(self, x : torch.Tensor, time_step : torch.Tensor):
        return super().forward(x, time_step).sample
    
class UNet1D(UNet1DModel):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def forward(self, x : torch.Tensor, time_step : torch.Tensor):
        return super().forward(x, time_step).sample
    
class UNet0D(UNet1DModel):
    def __init__(
        self,
        in_features : int,
        out_features : int,
        num_layers : int,
    ):
        down_block_types = ["DownBlock1D" for _ in range(num_layers)]
        up_block_types = ["UpBlock1D" for _ in range(num_layers)]
        block_out_channels = [32 for _ in range(num_layers)]

        super().__init__(
            in_channels=1,
            out_channels=1,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            )
        
        self.in_features_lin = torch.nn.Linear(in_features, 64)
        self.out_features_lin = torch.nn.Linear(64, out_features)

    def forward(self, x : torch.Tensor, time_step : torch.Tensor):
        x = self.in_features_lin(x).unsqueeze(1)
        x = super().forward(x, time_step).sample
        x = x.squeeze(1)
        x = self.out_features_lin(x)
        return x
    
if __name__ == "__main__":
    model = UNet0D(2, 2, 4)
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params)
    x = torch.randn(10, 2)
    time_step = torch.randn(10)
    out = model(x, time_step)
    print(out.shape)
