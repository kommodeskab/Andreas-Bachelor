from torch.utils.data import Dataset
import random

class OTMixDataset(Dataset):
    def __init__(
        self,
        dataset1 : Dataset,
        dataset2 : Dataset,
        random : bool = True,
    ):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.random = random
        self.len = max(len(dataset1), len(dataset2))
        
    def __len__(self):
        return self.len
    
    def _get_idx_for_dataset(self, idx, dataset):
        if not self.random:
            idx = idx % len(dataset)
        elif idx >= len(dataset):
            idx = random.randint(0, len(dataset) - 1)
        return idx
    
    def __getitem__(self, idx):
        d1_idx = self._get_idx_for_dataset(idx, self.dataset1)
        d2_idx = self._get_idx_for_dataset(idx, self.dataset2)
        return {
            "x0": self.dataset1[d1_idx],
            "xN": self.dataset2[d2_idx],
        }
        
if __name__ == "__main__":
    d1 = range(5)
    d2 = range(10)
    d = OTMixDataset(d1, d2, random=False)
    print(len(d))
    for i in range(20):
        print(d[i % len(d)])