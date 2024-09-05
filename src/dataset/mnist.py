from torch.utils.data import Dataset
from torchvision import datasets, transforms

class FilteredMNIST(Dataset):
    def __init__(
        self, 
        download : bool = False, 
        digit : int = 0,
        ):
        root = "data"
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))])
        self.mnist_dataset = datasets.MNIST(root=root, transform=transform, download=download)
        self.digit = digit
        self.indices = [i for i, label in enumerate(self.mnist_dataset.targets) if label == self.digit]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, _ = self.mnist_dataset[original_idx]
        return image
    
if __name__ == "__main__":
    dataset = FilteredMNIST(digit=3, download = True)
    image = dataset[0]
    print(len(dataset))
    print(image)
    print(image.shape)