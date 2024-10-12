import pytorch_lightning as pl
import torch
import torch.utils
from torch.utils.data import DataLoader, random_split, Dataset
from src.dataset.cacheloader import CacheDataLoader

class BaseDSBDM(pl.LightningDataModule):
    def __init__(
        self,
        start_dataset : Dataset,
        end_dataset : Dataset,
        cache_num_iters : int,
        train_val_split : float = 0.95,
        batch_size : int = 10,
        num_workers: int = 4,
        ):
        
        super().__init__()
        self.save_hyperparameters(ignore=["start_dataset", "end_dataset"])
        self.hparams["training_backward"] = True
        
        self.start_dataset = start_dataset
        self.end_dataset = end_dataset
        self.start_dataset_train, self.start_dataset_val = random_split(start_dataset, [train_val_split, 1 - train_val_split])
        self.end_dataset_train, self.end_dataset_val = random_split(end_dataset, [train_val_split, 1 - train_val_split])
        
        self.loader_kwargs = {
            "batch_size" : batch_size,
            "num_workers" : num_workers,
            "persistent_workers" : True,
            "drop_last" : True,
        }
    
    def train_dataloader(self):
        training_backward = self.hparams.training_backward
        dataset = self.start_dataset_train if training_backward else self.end_dataset_train
        return CacheDataLoader(dataset = dataset, cache_num_iters = self.hparams.cache_num_iters, shuffle = True, **self.loader_kwargs)
        
    def val_dataloader(self):
        return [
            DataLoader(dataset = self.start_dataset_val, shuffle = False, **self.loader_kwargs),
            DataLoader(dataset = self.end_dataset_val, shuffle = False, **self.loader_kwargs)
        ]