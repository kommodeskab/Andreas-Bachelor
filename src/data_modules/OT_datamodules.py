import pytorch_lightning as pl
import torch
import torch.utils
from torch.utils.data import DataLoader, random_split, Dataset
from src.dataset.distributions import StandardNormalDataset, Uniform2dDataset, Circle2dDataset
from src.dataset.mnist import FilteredMNIST
from src.dataset.datasettypes import RandomMixDataset

class StandardSchrodingerDM(pl.LightningDataModule):
    def __init__(
        self,
        start_dataset : Dataset,
        end_dataset : Dataset,
        train_val_split : float = 0.95,
        batch_size : int = 10,
        num_workers: int = 4,
        ):
        
        super().__init__()
        self.save_hyperparameters(ignore=["start_dataset", "end_dataset"])
                
        self.start_dataset_train, self.start_dataset_val = random_split(start_dataset, [train_val_split, 1 - train_val_split])
        self.end_dataset_train, self.end_dataset_val = random_split(end_dataset, [train_val_split, 1 - train_val_split])
        self.train_set = RandomMixDataset(self.start_dataset_train, self.end_dataset_train)
        self.loader_kwargs = {
            "batch_size" : batch_size,
            "num_workers" : num_workers,
            "persistent_workers" : True
        }
    
    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, **self.loader_kwargs)
        
    def val_dataloader(self):
        return [
            DataLoader(self.start_dataset_val, shuffle = False, **self.loader_kwargs),
            DataLoader(self.end_dataset_val, shuffle = False, **self.loader_kwargs)
            ]
        
class GaussianSchrodingerDM(StandardSchrodingerDM):
    def __init__(
        self,
        batch_size : int = 10,
        size : int = 1000,
    ):
        self.start_mu = torch.zeros(5)
        self.start_sigma = torch.ones(5) * 0.1

        start_dataset = StandardNormalDataset(
            mu = self.start_mu,
            sigma = self.start_sigma,
            size = size
        )

        self.end_mu = torch.ones(5)
        self.end_sigma = torch.ones(5) * 0.5

        end_dataset = StandardNormalDataset(
            mu = self.end_mu,
            sigma = self.end_sigma,
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
    ):
        start_dataset = FilteredMNIST(
            download = True, 
            digit = 1,
            )
        
        end_dataset = FilteredMNIST(
            download = True,
            digit = 7,
            )
        
        super().__init__(
            start_dataset = start_dataset, 
            end_dataset = end_dataset, 
            batch_size = batch_size
            )