import torch
import torch.nn as nn

class BaseBlock(nn.Module):
    def __init__(
        self,
        in_channels : int, 
        out_channels : int, 
        dropout : float = 0.2,
        ):
        super(BaseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.Identity()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return x

class DownBlock(BaseBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.2,
        ):
        super().__init__(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool2d(2)
    
class UpBlock(BaseBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.2,
        ):
        super().__init__(in_channels, out_channels, dropout)
        self.pool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

if __name__ == "__main__":
    x = torch.randn(1, 3, 32, 32)
    down_block = DownBlock(3, 64)
    up_block = UpBlock(3, 64)
    out = down_block(x)
    print(out.shape)
    out = up_block(x)
    print(out.shape)