import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

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
        self.values = self.values - 0.5
        
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
    
def plot_dataset(dataset, name = "plot"):
    data = dataset.values
    plt.scatter(data[:, 0], data[:, 1])
    plt.savefig(f"{name}.png")
    
if __name__ == "__main__":
    dataset = Circle2dDataset()
    plot_dataset(dataset, "circle")