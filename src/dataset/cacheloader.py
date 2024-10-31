from torch.utils.data import DataLoader

class CacheDataLoader(DataLoader):
    def __init__(self, cache_num_iters : int, max_cache_size : int, **kwargs):
        """
        A dataloader to be used with pytorch lightning modules that needs data caches
        The cache should only be updated every num_iters iterations
        Therefore, only provide a batch every num_iters iterations
        Otherwise, provide a 0
        Taking care of the batches and cache should happen in the training loop
        """
        super().__init__(**kwargs)
        self.cache_num_iters = cache_num_iters
        self.max_cache_size = max_cache_size

    def __iter__(self):
        data_iter = super().__iter__()
        for i, batch in enumerate(data_iter, start = 1):
            yield batch
            
            if i % self.max_cache_size == 0:
                for _ in range(self.cache_num_iters):
                    yield 0
    
    def __len__(self):
        return self.cache_num_iters * super().__len__()
                
if __name__ == "__main__":
    dataset = range(30)
    dataloader = CacheDataLoader(dataset = dataset, batch_size = 3, cache_num_iters = 2, max_cache_size = 4, drop_last = True, shuffle=True)
    for batch in dataloader:
        print(batch)
    for batch in dataloader:
        print(batch)