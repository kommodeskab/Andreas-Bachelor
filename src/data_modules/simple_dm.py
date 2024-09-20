from src.data_modules.base_dm import BaseDSBDM
from src.dataset.distributions import StandardNormalDataset, Uniform2dDataset, Circle2dDataset
import torch


class GaussianSchrodingerDM(BaseDSBDM):
    def __init__(
        self,
        size : int = 1000,
        **kwargs,
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
            **kwargs
            )
        
class Fun2dSchrodingerDM(BaseDSBDM):
    def __init__(
        self,
        size : int = 1000,
        **kwargs,
    ):
        start_dataset = Uniform2dDataset(size = size)
        end_dataset = Circle2dDataset(size = size)
        
        super().__init__(
            start_dataset = start_dataset, 
            end_dataset = end_dataset, 
            **kwargs
            )