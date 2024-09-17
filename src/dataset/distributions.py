import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random

class StandardNormalDataset(Dataset):
    def __init__(self, mu, sigma, size : int = 1000):
        self.size = size
        
        if isinstance(sigma, int):
            sigma = torch.ones_like(mu) * sigma
        
        self.values = torch.distributions.Normal(mu, sigma).sample((size, ))
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.values[index]
    
class Uniform2dDataset(Dataset):
    def __init__(self, size = 1000):
        self.size = size
        self.values = torch.rand(size, 2)
        # center the data around 0
        self.values = self.values * 2 - 1
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.values[index]
    
class Circle2dDataset(Dataset):
    def __init__(self, r = 1, size = 1000):
        self.size = size
        self.r = torch.sqrt(torch.rand(size)) * r
        self.theta = torch.rand(size) * 2 * 3.1415
        self.values = torch.stack([self.r * torch.cos(self.theta), self.r * torch.sin(self.theta)], dim = 1)
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.values[index]
    
class ChessBoard2dDataset(Dataset):
    def __init__(self, size : int = 1000):
        self.size = size
        super().__init__()
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        x_square = random.randint(0, 7)
        if x_square % 2 == 0:
            y_square = random.sample([1, 3, 5, 7], 1)[0]
        else:
            y_square = random.sample([0, 2, 4, 6], 1)[0]
            
        x_val, y_val = x_square + random.random(), y_square + random.random()
        x_val, y_val = x_val / 4 - 1, y_val / 4 - 1
        return torch.tensor([x_val, y_val])
    
def plot_dataset(dataset, name = "plot"):
    data = [dataset[i] for i in range(len(dataset))]
    data = torch.stack(data).numpy()
    print(data)
    plt.scatter(data[:, 0], data[:, 1])
    plt.savefig(f"{name}.png")
    
if __name__ == "__main__":
    dataset = ChessBoard2dDataset()
    plot_dataset(dataset, "chess")