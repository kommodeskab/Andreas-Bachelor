from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor, Resize, Compose
import torch
import matplotlib.pyplot as plt

class CelebADataset(Dataset):
    def __init__(self, img_size : int = 32, download : bool = False):
        super().__init__()
        
        transform = Compose([
                    Resize((img_size, img_size)),
                    ToTensor(),
                ])
        
        self.celeba = CelebA(
            root="data",
            split="all",
            download=download,
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
    dataset = FilteredByAttrCelebA(attr=31, on_or_off=0)
    fig, axs = plt.subplots(1, 10, figsize=(20, 2))
    for i, ax in enumerate(axs):
        ax.imshow(dataset[i].permute(1, 2, 0).squeeze() * 0.5 + 0.5)
    plt.savefig("data/samples/filtered_celeba_no_smiling.png")

        