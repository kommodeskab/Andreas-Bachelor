from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class EMNIST(Dataset):
    def __init__(self, split : str):
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((32, 32)),
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
        image, _ = self.emnist_dataset[idx]
        image = transforms.functional.rotate(image, 90)
        image = transforms.functional.vflip(image)
        return image
     
if __name__ == "__main__":
    dataset = EMNIST("digits")
    fig, ax = plt.subplots(5, 5)
    for i in range(5):
        for j in range(5):
            ax[i, j].imshow(dataset[i * 5 + j].squeeze(), cmap="gray")
            ax[i, j].axis("off")
    plt.show()