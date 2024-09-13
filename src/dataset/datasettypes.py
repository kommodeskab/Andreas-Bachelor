from torch.utils.data import Dataset
import random

class RandomMixDataset(Dataset):
    def __init__(
        self,
        dataset1 : Dataset,
        dataset2 : Dataset,
    ):
        """
        Given two datasets, this class will return a tuple of two samples from the two datasets.
        The samples are chosen randomly from the two datasets without resampling.
        
        If the datasets are of different lengths, the new mix dataset will have the length of the shorter dataset.
        The samples from the longer dataset are then chosen randomly.
        
        """
        
        self.long_dataset, self.short_dataset = (dataset1, dataset2) if len(dataset1) > len(dataset2) else (dataset2, dataset1)
        
        self.length = len(self.short_dataset)
        long_indices = list(range(len(self.long_dataset)))
        random.shuffle(long_indices)
        self.long_indices = long_indices
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        short_dataset_sample = self.short_dataset[idx]
        long_dataset_sample = self.long_dataset[self.long_indices[idx]]
        return short_dataset_sample, long_dataset_sample
        
if __name__ == "__main__":
    d1 = range(5)
    d2 = range(10)
    d = RandomMixDataset(d1, d2)
    print(len(d))
    for i in range(len(d)):
        print(d[i])