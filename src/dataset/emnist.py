from torchvision import datasets, transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

class EMNIST(Dataset):
    def __init__(self, split : str, img_size : int = 32):
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((img_size, img_size)),
                ])
        self.emnist_dataset = datasets.EMNIST(
            root="data",
            split=split,
            download=True,
            transform=transform,
        )
        super().__init__()
        
    def __len__(self):
        return len(self.emnist_dataset)
    
    def __getitem__(self, idx):
        image, label = self.emnist_dataset[idx]
        image = transforms.functional.rotate(image, 90)
        image = transforms.functional.vflip(image)
        return image * 2 - 1, label
    
class EMNISTNoLabel(EMNIST):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __getitem__(self, idx):
        image, _ = super().__getitem__(idx)
        return image
    
class FilteredMNIST(Dataset):
    def __init__(self, digit : int, img_size : int = 32):
        super().__init__()
        self.original_dataset = EMNIST(split="digits", img_size=img_size)
        targets = datasets.EMNIST(root="data", split="digits", download=True).targets
        self.indices = [i for i, label in enumerate(targets) if label == digit]

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, _ = self.original_dataset[original_idx]
        return image
     
if __name__ == "__main__":
    dataset = FilteredMNIST(digit=2)
    fig, axs = plt.subplots(1, 5)
    for i, ax in enumerate(axs):
        ax.imshow(dataset[i][0].squeeze() * 0.5 + 0.5, cmap="gray")
    plt.savefig("filtered_mnist.png")