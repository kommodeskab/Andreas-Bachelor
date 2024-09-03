import pytorch_lightning as pl
import torch
import torch.utils
from torch.utils.data import DataLoader, random_split, Dataset
from src.dataset.distributions import StandardNormalDataset, Uniform2dDataset, Circle2dDataset
from src.dataset.mnist import FilteredMNIST

class StandardSchrodingerDM(pl.LightningDataModule):
    def __init__(
        self,
        start_dataset : Dataset,
        end_dataset : Dataset,
        train_val_split : float = 0.9,
        training_backward : bool = True,
        batch_size : int = 10,
        num_workers: int = 4,
        ):
        
        super().__init__()
        self.save_hyperparameters(ignore=["start_dataset", "end_dataset"])
                
        self.start_dataset_train, self.start_dataset_val = random_split(start_dataset, [train_val_split, 1 - train_val_split])
        self.end_dataset_train, self.end_dataset_val = random_split(end_dataset, [train_val_split, 1 - train_val_split])
    
    def train_dataloader(self):
        if self.hparams.training_backward:
            return DataLoader(self.start_dataset_train, batch_size = self.hparams.batch_size, shuffle = True, num_workers = self.hparams.num_workers, persistent_workers=True)
        else:
            return DataLoader(self.end_dataset_train, batch_size = self.hparams.batch_size, shuffle = True, num_workers = self.hparams.num_workers, persistent_workers=True)
        
    def val_dataloader(self):
        return [
            DataLoader(self.start_dataset_val, batch_size = self.hparams.batch_size, num_workers = self.hparams.num_workers, persistent_workers=True),
            DataLoader(self.end_dataset_val, batch_size = self.hparams.batch_size, num_workers = self.hparams.num_workers, persistent_workers=True)
        ]
        
class GaussianSchrodingerDM(StandardSchrodingerDM):
    def __init__(
        self,
        batch_size : int = 10,
        size : int = 1000,
    ):
        start_dataset = StandardNormalDataset(
            mu = torch.tensor([0.0, 0.0]),
            sigma = torch.tensor([0.1, 0.1]),
            size = size
        )
        end_dataset = StandardNormalDataset(
            mu = torch.tensor([1.0, 1.0]),
            sigma = torch.tensor([0.1, 0.1]),
            size = size
        )
        
        super().__init__(
            start_dataset = start_dataset, 
            end_dataset = end_dataset, 
            batch_size = batch_size
            )
        
class Fun2dSchrodingerDM(StandardSchrodingerDM):
    def __init__(
        self,
        batch_size : int = 10,
        size : int = 1000,
    ):
        start_dataset = Uniform2dDataset(size = size)
        end_dataset = Circle2dDataset(size = size)
        
        super().__init__(
            start_dataset = start_dataset, 
            end_dataset = end_dataset, 
            batch_size = batch_size
            )

class OneAndSevenSchrodingerDM(StandardSchrodingerDM):
    def __init__(
        self,
        batch_size : int = 10,
        size : int | None = None
    ):
        start_dataset = FilteredMNIST(
            download = True, 
            digit = 1,
            size = size
            )
        end_dataset = FilteredMNIST(
            download = True,
            digit = 7,
            size = size
            )
        
        super().__init__(
            start_dataset = start_dataset, 
            end_dataset = end_dataset, 
            batch_size = batch_size
            )