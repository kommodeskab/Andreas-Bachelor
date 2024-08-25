from src.data_modules.basedatamodule import BaseDataModule
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

class MNISTDataModule(BaseDataModule):
    def __init__(
        self,
        batch_size : int = 32,
        num_workers : int = 4,
        ):
        super().__init__(batch_size, num_workers)
        self.save_hyperparameters()
        
        self.train_dataset = MNIST(
            root = "data",
            train = True,
            download = True,
            transform = ToTensor(),
        )
        
        self.val_dataset = MNIST(
            root = "data",
            train = False,
            download = True,
            transform = ToTensor(),
        )