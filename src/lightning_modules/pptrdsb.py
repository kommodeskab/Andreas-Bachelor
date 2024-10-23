from src.lightning_modules.reparameterized_dsb import TRDSB
from torch import Tensor
from torch.utils.data import DataLoader
import torch

class PPTRDSB(TRDSB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, initial_forward_sampling = None)
    
    def _get_pseudo_xN(self, x0 : Tensor) -> Tensor:
        """
        Given some batched x0, return a "pseudo" terminal point
        For example, add noise, or find the closest point in the end dataset, etc.
        """
        raise NotImplementedError
    
    def forward_call(self, xk : Tensor, k : Tensor) -> Tensor:        
        if self.hparams.training_backward and self.hparams.DSB_iteration == 1:
            if k[0].item() == 0:
                self.pseudo_xN = self._get_pseudo_xN(xk)
                
            return self.pseudo_xN
        
        return super().forward_call(xk, k)
    
class PPTRDSBClosestPoint(PPTRDSB):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _get_pseudo_xN(self, x0 : Tensor) -> Tensor:
        end_dataset = self.trainer.datamodule.end_dataset_train
        num_samples = min(3_000, len(end_dataset))
        end_dataset_loader = DataLoader(end_dataset, batch_size=num_samples, num_workers=4)
        end_dataset_tensor : Tensor = next(iter(end_dataset_loader)).to(self.device)
        
        distances = torch.cdist(x0.flatten(1), end_dataset_tensor.flatten(1))
        closest_indices = torch.argmin(distances, dim = 1)
        xN_pred = end_dataset_tensor[closest_indices]
        return xN_pred