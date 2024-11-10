from torch.utils.data import Dataset
import hashlib
from torchvision import transforms

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    @property
    def unique_identifier(self):
        attr_str = str(vars(self))
        return hashlib.md5(attr_str.encode()).hexdigest()
    
class ImageDataset(BaseDataset):
    def __init__(
        self,
        dataset : Dataset,
        img_size : int,
        augment : bool = False,
        size_multiplier : int = 1,
    ):
        super().__init__()
        self.dataset = dataset
        self.size_multiplier = size_multiplier
        
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop((img_size, img_size), scale=(0.9, 1.0)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            
    def __len__(self):
        return int(len(self.dataset) * self.size_multiplier)
    
    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        img = self.dataset[idx]
        img = self.transform(img).clamp(-1, 1)
        return img