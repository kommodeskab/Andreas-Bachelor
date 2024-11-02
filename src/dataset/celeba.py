from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor, Resize, Compose
import torch
import matplotlib.pyplot as plt
from .basedataset import BaseDataset

class CelebADataset(BaseDataset):
    def __init__(self, img_size : int = 32):
        super().__init__()
        
        transform = Compose([
                    Resize((img_size, img_size)),
                    ToTensor(),
                ])
        
        self.celeba = CelebA(
            root="data",
            split="all",
            download=True,
            transform=transform,
            target_type="attr",
        )
    
    @property
    def attr(self):
        return self.celeba.attr
        
    def __len__(self):
        return len(self.celeba)
    
    def __getitem__(self, idx):
        return self.celeba[idx]

class CelebANoLabel(CelebADataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        img = img * 2 - 1
        return img
    
class FilteredByAttrCelebA(CelebANoLabel):
    def __init__(self, attr : int, on_or_off : bool, img_size : int = 32):
        super().__init__(img_size = img_size)
        mask = self.attr[:, attr] == on_or_off
        self.indices = torch.arange(len(self.celeba))[mask]
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        return super().__getitem__(idx)
    
if __name__ == "__main__":
    dataset = FilteredByAttrCelebA(attr=20, on_or_off=1)
    print(len(dataset))

        