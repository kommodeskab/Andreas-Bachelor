from torch.utils.data import DataLoader

class CacheDataLoader(DataLoader):
    def __init__(self, cache_num_iters : int, **kwargs):
        """
        A dataloader to be used with pytorch lightning modules that needs data caches
        The cache should only be updated every num_iters iterations
        Therefore, only provide a batch every num_iters iterations
        Otherwise, provide a 0
        Taking care of the batches and cache should happen in the training loop
        """
        super().__init__(**kwargs)
        self.cache_num_iters = cache_num_iters

    def __iter__(self):
        for batch in super().__iter__():
            yield batch
            
            for _ in range(self.cache_num_iters - 1):
                yield 0
    
    def __len__(self):
        return super().__len__() * self.cache_num_iters
    
class CacheDataLoader2(DataLoader):
    def __init__(self, cache_num_iters : int, max_cache_size : int, **kwargs):
        super().__init__(**kwargs)
        assert cache_num_iters >= max_cache_size, f"{cache_num_iters = } should be greater than {max_cache_size = }"
        self.cache_num_iters = cache_num_iters
        self.max_cache_size = max_cache_size
        self.original_number_of_batches = super().__len__()
        assert self.original_number_of_batches >= self.max_cache_size, (
            f"{self.original_number_of_batches = } should be greater than or equal to {self.max_cache_size = }. Otherwise, the same batch will be in the cache multiple times => inefficient."
        )

    def __iter__(self):
        data_iter = super().__iter__()
        for i, batch in enumerate(data_iter, start = 1):
            yield batch
            
            if i % self.max_cache_size == 0:
                for _ in range(self.cache_num_iters - self.max_cache_size):
                    yield 0
    
    def __len__(self):
        num_repeats = self.original_number_of_batches // self.max_cache_size
        return num_repeats * self.cache_num_iters
                
if __name__ == "__main__":
    dataset = range(20)
    dataloader = CacheDataLoader(dataset = dataset, batch_size = 3, cache_num_iters = 5, drop_last = True, shuffle=True)
    for batch in dataloader:
        print(batch)