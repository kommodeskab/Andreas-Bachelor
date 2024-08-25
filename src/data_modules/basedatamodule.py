import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split

class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size : int = 32,
        num_workers : int = 4,
        dataset : Dataset | None = None,
        train_val_frac : float = 0.8,
        ):
        super().__init__()
        self.save_hyperparameters(ignore=("dataset"))
        
        if dataset is not None:
            self.train_dataset, self.val_dataset = random_split(
                dataset,
                [train_val_frac, 1 - train_val_frac],
            )
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.hparams.batch_size,
            shuffle = True,
            num_workers = self.hparams.num_workers,
            persistent_workers = True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.hparams.batch_size,
            shuffle = False,
            num_workers = self.hparams.num_workers,
            persistent_workers = True,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.hparams.batch_size,
            shuffle = False,
            num_workers = self.hparams.num_workers,
            persistent_workers = True,
        )