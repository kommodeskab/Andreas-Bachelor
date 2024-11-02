import os
from torchvision import transforms
from PIL import Image
import torch
from src.dataset.basedataset import BaseDataset

class AFHQ(BaseDataset):
    """
    Crawls the AFHQ dataset
    The dataset is located in the following structure:
    - data
        - afhq
            - train
                - cat
                - dog
                - wild
    """
    def __init__(self, split : str, img_size : int = 128, train : bool = True):
        train = "train" if train else "val"
        root = os.path.join(f"data/afhq/{train}", split)
        self.files = [os.path.join(root, file) for file in os.listdir(root)]
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
        ])
            
        
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx) -> torch.Tensor:
        img = Image.open(self.files[idx])
        img = self.transform(img)
        return img
    
if __name__ == "__main__":
    dataset = AFHQ(split="cat")
    print(len(dataset))
    dataset = AFHQ(split="dog")
    print(len(dataset))