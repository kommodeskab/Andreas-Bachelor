from src.networks.basetorchmodule import BaseTorchModule
from src.networks.utils import UpBlock, DownBlock, BaseBlock
from functools import reduce
import torch
import torch.nn as nn

def prod_of_tuple(t):
    return reduce(lambda x, y: x * y, t)

class SimpleFFN(BaseTorchModule):
    def __init__(
        self,
        in_features : int,
        out_features : int,   
        hidden_layer_size : int = 128
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.fc1 = nn.Linear(in_features, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
class SimpleEncoderForImages(SimpleFFN):
    def __init__(
        self,
        height : int,
        width : int, 
        channels : int,
        out_features : int,
        hidden_layer_size : int = 128
        ):
        in_features = height * width * channels
        super().__init__(in_features, out_features, hidden_layer_size)

        self.mu_lin = nn.Linear(out_features, out_features)
        self.logvar_lin = nn.Linear(out_features, out_features)
        
    def forward(self, x):
        size = x.shape
        x = x.view(size[0], -1)
        x = super().forward(x)
        x = nn.functional.relu(x)
        mu = self.mu_lin(x)
        logvar = self.logvar_lin(x)
        return mu, logvar
        
class SimpleDecoderForImages(SimpleFFN):
    def __init__(
        self,
        in_features : int,
        height : int,
        width : int, 
        channels : int,
        hidden_layer_size : int = 128
        ):
        out_features = height * width * channels
        super().__init__(in_features, out_features, hidden_layer_size)
        
        self.height = height
        self.width = width
        self.channels = channels
        
    def forward(self, z):
        x = super().forward(z)
        return x.view(-1, self.channels, self.height, self.width)
    
class ImageEncoder(BaseTorchModule):
    def __init__(
        self,
        height : int,
        width : int, 
        channels : int,
        latent_dim : int,
        channels_list : list[int] = [32, 64, 128, 256]
        ):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        in_channels = channels
        for out_channels in channels_list:
            self.blocks.append(DownBlock(in_channels, out_channels))
            in_channels = out_channels
            
        final_height = height // 2 ** len(channels_list)
        final_width = width // 2 ** len(channels_list)
        final_size = channels_list[-1] * final_height * final_width
        
        self.mu_lin = nn.Linear(final_size, latent_dim)
        self.logvar_lin = nn.Linear(final_size, latent_dim)
        
    def forward(self, x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x)
        
        x = x.view(x.size(0), -1)

        mu = self.mu_lin(x)
        logvar = self.logvar_lin(x)
        
        return mu, logvar
    
class ImageDecoder(BaseTorchModule):
    def __init__(
        self,
        height : int,
        width : int, 
        channels : int,
        latent_dim : int,
        channels_list : list[int] = [256, 128, 64, 32]
        ):
        super().__init__()
        
        final_height = height // 2 ** len(channels_list)
        final_width = width // 2 ** len(channels_list)
        self.final_size = (channels_list[0], final_height, final_width)
        self.from_latent = nn.Linear(latent_dim, prod_of_tuple(self.final_size))
        
        self.blocks = nn.ModuleList()
        channels_list += [channels]
        in_channels = channels_list[0]
        for out_channels in channels_list[1:]:
            self.blocks.append(UpBlock(in_channels, out_channels))
            in_channels = out_channels
        
        self.final_block = BaseBlock(channels, channels)
        self.final_conv = nn.Conv2d(channels, channels, kernel_size = 1)
            
    def forward(self, z):
        x = self.from_latent(z)
        x = x.view(-1, *self.final_size)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.final_block(x)
        x = self.final_conv(x)
            
        return x