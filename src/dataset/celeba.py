from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor, Resize, Compose
import torch

class CelebADataset(Dataset):
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
    
class FilteredByAttrCelebA(CelebADataset):
    def __init__(self, attr : int, on_or_off : bool, img_size : int = 32):
        super().__init__(img_size = img_size)
        mask = self.attr[:, attr] == on_or_off
        self.indices = torch.arange(len(self.celeba))[mask]
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        img, _ = self.celeba[idx]
        img = img * 2 - 1
        return img
        