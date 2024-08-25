from src.networks.basetorchmodule import BaseTorchModule
import torch

class SimpleFFN(BaseTorchModule):
    def __init__(
        self,
        in_features : int,
        out_features : int,   
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.fc1 = torch.nn.Linear(in_features, 128)
        self.fc2 = torch.nn.Linear(128, out_features)
        self.activation = torch.nn.ReLU()
        
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
        ):
        in_features = height * width * channels
        super().__init__(in_features, out_features)
        
    def forward(self, x):
        size = x.shape
        x = x.view(size[0], -1)
        return super().forward(x)
        
class SimpleDecoderForImages(SimpleFFN):
    def __init__(
        self,
        in_features : int,
        height : int,
        width : int, 
        channels : int,
        ):
        out_features = height * width * channels
        super().__init__(in_features, out_features)
        
        self.height = height
        self.width = width
        self.channels = channels
        
    def forward(self, z):
        x = super().forward(z)
        return x.view(-1, self.channels, self.height, self.width)