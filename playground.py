import random
import torch
from torch import Tensor
from typing import List, Tuple

class Cache:
    def __init__(self, max_size : int):
        self.cache = []
        self.max_size = max_size
        
    def add(self, sample: Tuple[Tensor, Tensor]):
        """
        Add a sample to the cache.
        """
        if len(self) >= self.max_size:
            self.cache.pop(0)
            
        self.cache.append(sample)
        
    def sample(self) -> Tuple[Tensor, Tensor]:
        """
        Randomly sample a sample from the cache. The sample is removed from the cache.
        """
        randint = random.randint(0, len(self.cache) - 1)
        return self.cache[randint]
        
    def __len__(self):
        return len(self.cache)
    
    def __str__(self):
        return self.cache.__str__()

cache = Cache(max_size = 10)
for i in range(20):
    cache.add(random.randint(0, 10))
    print(cache)
    
sample = cache.sample()
print(sample)
print(len(cache))
print(cache)