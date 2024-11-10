from pytorch_lightning.callbacks import Callback
import torch
from src.callbacks.utils import get_batch_from_dataset
from src.lightning_modules.ddpm import DDPM
import wandb
import matplotlib.pyplot as plt
from src.callbacks.plot_functions import get_image_fig, get_grid_fig

class DiffusionCallback(Callback):
    def __init__(self):
        super().__init__()
        
    def on_train_start(self, trainer, pl_module):
        dataset = trainer.datamodule.train_dataset
        batch = get_batch_from_dataset(dataset, 1)[0]
        self.sample_shape = batch.shape
        
    def on_validation_end(self, trainer, pl_module : DDPM) -> None:
        noise = torch.randn(16, *self.sample_shape).to(pl_module.device)
        trajectory = pl_module.sample(noise, return_trajectory=True, clamp=True).cpu()
        trajectory = (trajectory + 1) / 2
        fig_1 = get_image_fig(trajectory)
        
        left_images = noise.permute(0, 2, 3, 1).cpu()
        left_images = (left_images + 1) / 2
        left_images = left_images.clamp(0, 1)
        right_images = trajectory[-1].permute(0, 2, 3, 1).cpu()
        fig_2 = get_grid_fig(left_images, right_images, num_rows=4)
        
        pl_module.logger.log_image(
            "Trajectory",
            [wandb.Image(fig_1)],
            step = trainer.global_step
        )
        pl_module.logger.log_image(
            "Samples",
            [wandb.Image(fig_2)],
            step = trainer.global_step
        )
        plt.close(fig_1)
        plt.close(fig_2)