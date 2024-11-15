import pytorch_lightning as pl
import torch
import torch.utils
from torch.utils.data import DataLoader, random_split, Dataset
from src.dataset.cacheloader import CacheDataLoader

class BaseDM(pl.LightningDataModule):
    def __init__(
        self,
        dataset : Dataset,
        val_dataset : Dataset = None,
        train_val_split : float = 0.95,
        batch_size : int = 10,
        num_workers: int = 4,
        ):
        """
        A base data module for datasets. 
        It takes a dataset and splits into train and validation (if val_dataset is None).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["dataset", "val_dataset"])
        
        self.dataset = dataset
        
        if val_dataset is None:
            self.train_dataset, self.val_dataset = random_split(dataset, [train_val_split, 1 - train_val_split])
        else:
            self.train_dataset, self.val_dataset = dataset, val_dataset
        
    def train_dataloader(self):
        return DataLoader(
            dataset = self.train_dataset, 
            shuffle = True, 
            drop_last=True,
            persistent_workers=True,
            batch_size = self.hparams.batch_size, 
            num_workers = self.hparams.num_workers
            )
        
    def val_dataloader(self):
        return DataLoader(
            dataset = self.val_dataset, 
            shuffle = False, 
            drop_last=True,
            persistent_workers=True,
            batch_size = self.hparams.batch_size, 
            num_workers = self.hparams.num_workers
            )

class BaseDSBDM(pl.LightningDataModule):
    def __init__(
        self,
        start_dataset : Dataset,
        end_dataset : Dataset,
        start_dataset_val : Dataset = None,
        end_dataset_val : Dataset = None,
        train_val_split : float = 0.95,
        batch_size : int = 10,
        num_workers: int = 4,
        ):
        """
        A special datamodule for the DSB algorithm. 
        It takes two datasets, one for the start and one for the end, and splits them into train and validation.
        It returns end_dataset when training forward and start_dataset when training backward.
        Also uses the special CacheDataLoader for the cache implementation in the DSB algorithm.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["start_dataset", "end_dataset", "start_dataset_val", "end_dataset_val"])
        
        self.start_dataset = start_dataset
        self.end_dataset = end_dataset
        
        if start_dataset_val is None:
            self.start_dataset_train, self.start_dataset_val = random_split(start_dataset, [train_val_split, 1 - train_val_split])
        else:
            self.start_dataset_train, self.start_dataset_val = start_dataset, start_dataset_val
        
        if end_dataset_val is None:
            self.end_dataset_train, self.end_dataset_val = random_split(end_dataset, [train_val_split, 1 - train_val_split])
        else:
            self.end_dataset_train, self.end_dataset_val = end_dataset, end_dataset_val
        
        self.loader_kwargs = {
            "batch_size" : batch_size,
            "num_workers" : num_workers,
            "persistent_workers" : True,
            "drop_last" : True,
        }
    
    def train_dataloader(self):
        training_backward = self.hparams.training_backward
        dataset = self.start_dataset_train if training_backward else self.end_dataset_train
        return DataLoader(
            dataset = dataset, 
            shuffle = True,
            **self.loader_kwargs
            )
        
    def val_dataloader(self):
        return [
            DataLoader(dataset = self.start_dataset_val, shuffle = False, **self.loader_kwargs),
            DataLoader(dataset = self.end_dataset_val, shuffle = False, **self.loader_kwargs)
        ]