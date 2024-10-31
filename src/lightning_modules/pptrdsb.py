from src.lightning_modules.reparameterized_dsb import TRDSB
from torch import Tensor
from torch.utils.data import DataLoader
import torch
from src.dataset.audio import Noise
import random

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
    def __init__(self, num_samples_to_check : int = 5_000, **kwargs):
        self.num_samples_to_check = num_samples_to_check
        super().__init__(**kwargs)
        
    def _get_pseudo_xN(self, x0 : Tensor) -> Tensor:
        end_dataset = self.trainer.datamodule.end_dataset_train
        num_samples = min(self.num_samples_to_check, len(end_dataset))
        end_dataset_loader = DataLoader(end_dataset, batch_size=num_samples, num_workers=4)
        end_dataset_tensor : Tensor = next(iter(end_dataset_loader)).to(self.device)
        
        distances = torch.cdist(x0.flatten(1), end_dataset_tensor.flatten(1))
        closest_indices = torch.argmin(distances, dim = 1)
        xN_pred = end_dataset_tensor[closest_indices]
        return xN_pred
    
class PPTRDSBNoisyAudio(PPTRDSB):
    def __init__(self, sample_rate, audio_length):
        self.noise = Noise(
            sample_rate = sample_rate,
            audio_length = audio_length
        )
        super().__init__()
        
    def _get_pseudo_xN(self, x0):
        batch_size = x0.shape[0]
        # get batch_size number of noise samples
        noise = [self.noise[random.randint(0, len(self.noise) - 1)] for _ in range(batch_size)]
        noise = torch.stack(noise).to(self.device)
        alpha = random.uniform(0.5, 1.5)
        return x0 + alpha * noise