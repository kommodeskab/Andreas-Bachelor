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

    def forward(self, x, time_encoding):
        x_enc = self.x_encoder(x)
        x_enc = self.activation(x_enc)

        time_enc = self.time_encoder(time_encoding)
        time_enc = self.activation(time_enc)

        out = x_enc + time_enc
        
        return out

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

    def forward(self, x : torch.Tensor, t : torch.Tensor):
        t = t.unsqueeze(1)
        time_encoding = sinusoidal_encoding(t, self.time_encoding_size, self.exponential_base)
        for block in self.blocks:
            x = block(x, time_encoding)
        x = self.final_layer(x)
        return x