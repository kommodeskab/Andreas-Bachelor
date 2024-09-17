from torchvision import datasets, transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class EMNIST(Dataset):
    def __init__(self, split : str):
        transform = transforms.Compose([
                    lambda img: transforms.functional.rotate(img, -90),
                    lambda img: transforms.functional.hflip(img),
                    transforms.ToTensor(),
                    transforms.Resize((32, 32)),
                ])
        self.emnist_dataset = datasets.EMNIST(
            root="data",
            split=split,
            download=True,
            transform=transform,
        )
        
    def __len__(self):
        return len(self.emnist_dataset)
    
    def __getitem__(self, idx):
        image, _ = self.emnist_dataset[idx]
        return image
     
if __name__ == "__main__":
    dataset = EMNIST()
    print(len(dataset))
    print(dataset[0].shape)
    fig, ax = plt.subplots(5, 5)
    for i in range(5):
        for j in range(5):
            ax[i, j].imshow(dataset[i * 5 + j].squeeze(), cmap="gray")
    plt.show()