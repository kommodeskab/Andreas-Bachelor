from src.lightning_modules.reparameterized_dsb import TRDSB
from src.dataset.emnist import EMNISTNoLabel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dsb = TRDSB(forward_model=None, backward_model=None, optimizer=None, scheduler=None, max_gamma=0.1, min_gamma=0.001, num_steps=100, initial_forward_sampling="ornstein_uhlenbeck")
dataset = EMNISTNoLabel(split="letters", img_size=32)
batch = next(iter(DataLoader(dataset, batch_size=100)))
sampled = dsb.sample(batch, forward=True, return_trajectory=False)
sampled = sampled.flatten(1)
print(sampled.mean(1))