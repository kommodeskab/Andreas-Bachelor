from torch.utils.data import DataLoader

class CacheDataLoader(DataLoader):
    def __init__(self, num_iters : int, **kwargs):
        """
        A dataloader to be used with pytorch lightning modules that needs data caches
        The cache should only be updated every num_iters iterations
        Therefore, only provide a batch every num_iters iterations
        Otherwise, provide a 0
        Taking care of the batches and cache should happen in the training loop
        """
        super().__init__(**kwargs)
        self.num_iters = num_iters

    def __iter__(self):
        data_iter = super().__iter__()
        for batch in data_iter:
            yield batch
            
            for _ in range(self.num_iters - 1):
                yield 0
    
    def __len__(self):
        return self.num_iters * super().__len__()
                
if __name__ == "__main__":
    dataset = range(10)
    dataloader = CacheDataLoader(dataset = dataset, batch_size = 3, num_iters = 2, drop_last = True)
    for batch in dataloader:
        print(batch)
    for batch in dataloader:
        print(batch)