import torch
import torch.nn as nn
import math
from src.networks.basetorchmodule import BaseTorchModule

def sinusoidal_encoding(tensor, enc_size, exponential_base = 10000.0):
    device = tensor.device
    batch_size = tensor.size(0)
    
    position = torch.arange(0, enc_size, dtype=torch.float, device = device).unsqueeze(0)
    position = position.repeat(batch_size, 1)
    
    div_term = torch.exp(torch.arange(0, enc_size, 2, device = device).float() * (-math.log(exponential_base) / enc_size))
    
    position[:, 0::2] = torch.sin(tensor * div_term)
    position[:, 1::2] = torch.cos(tensor * div_term)

    return position

class FFN(nn.Module):
    def __init__(
        self, 
        in_features : int, 
        out_features : int,
        num_layers : int = 2,
        hidden_size : int = 64
    ):
        super().__init__()
        
        layers = [nn.Linear(in_features, hidden_size), nn.ReLU()]
        
        for _ in range(num_layers - 2):
            layers.append([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
            
        layers.append(nn.Linear(hidden_size, out_features))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
class TimeEncodingBlock(nn.Module):
    def __init__(
        self,
        in_features : int,
        out_features : int,
        time_encoding_size : int,
    ):
        super().__init__()
        self.x_encoder = nn.Linear(in_features, out_features)
        self.time_encoder = nn.Linear(time_encoding_size, out_features)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, time_encoding):
        x_enc = self.x_encoder(x)
        time_enc = self.time_encoder(time_encoding)
        x = x_enc + time_enc
        x = self.activation(x)
        x = self.dropout(x)
        return x
    
class FFNWithTimeEncoding(BaseTorchModule):
    def __init__(
        self,
        in_features : int,
        out_features : int,
        num_layers : int = 2,
        hidden_size : int = 64,
        time_encoding_size : int = 8,
        exponential_base : float = 10000.0
    ):
        super().__init__()
        self.time_encoding_size = time_encoding_size
        self.exponential_base = exponential_base
        sizes = [in_features] + [hidden_size] * (num_layers - 1)
        self.blocks = nn.ModuleList(
            [TimeEncodingBlock(sizes[i], sizes[i + 1], time_encoding_size) for i in range(len(sizes) - 1)]
            )
        self.final_layer = nn.Linear(hidden_size, out_features)
        
    def forward(self, x, t):
        time_encoding = sinusoidal_encoding(t, self.time_encoding_size, self.exponential_base)
        for block in self.blocks:
            x = block(x, time_encoding)
        x = self.final_layer(x)
        return x
    
class FFNWithTimeEncodingForImages(FFNWithTimeEncoding):
    def __init__(
        self, 
        in_channels : int,
        height : int,
        width : int,
        num_layers : int = 2,
        hidden_size : int = 64,
        time_encoding_size : int = 128,
        exponential_base : float = 10000.0
    ):
        in_features = in_channels * height * width
        out_features = in_channels * height * width
        super().__init__(
            in_features, 
            out_features, 
            num_layers, 
            hidden_size, 
            time_encoding_size, 
            exponential_base
        )
        
    def forward(self, x, t):
        size = x.size()
        x = x.view(size[0], -1)
        x = super().forward(x, t)
        x = x.view(size)
        return x
        
if __name__ == "__main__":
    x = torch.randn(10, 3, 32, 32)
    t = torch.randn(10, 1)
    model = FFNWithTimeEncodingForImages(3, 32, 32)
    y = model(x, t)
    print(y.size())
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")