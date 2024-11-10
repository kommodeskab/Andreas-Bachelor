from src.callbacks.plot_functions import get_grid_fig, get_image_fig
from src.lightning_modules.schrodinger_bridge import StandardDSB
from src.lightning_modules.reparameterized_dsb import TRDSB, FRDSB
from typing import Union
import pytorch_lightning as pl
from src.callbacks.utils import get_batch_from_dataset
import wandb
import matplotlib.pyplot as plt
from cleanfid import fid
import os
import torch
import shutil
import wandb
import uuid

class MarginalDistributionsImagesCB(pl.Callback):
    def __init__(self, num_rows : int = 5):
        super().__init__()
        self.num_rows = num_rows

    def on_train_start(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        x0 = get_batch_from_dataset(trainer.datamodule.start_dataset_val, self.num_rows ** 2, shuffle=True).to(pl_module.device)
        xN = get_batch_from_dataset(trainer.datamodule.end_dataset_val, self.num_rows ** 2, shuffle=True).to(pl_module.device)
        x0 = (x0 + 1) / 2
        xN = (xN + 1) / 2

        x0_fig, axs = plt.subplots(self.num_rows, self.num_rows, figsize=(20, 20))
        cmap = "gray" if x0.size(1) == 1 else None
        for i in range(self.num_rows ** 2):
            row, col = divmod(i, self.num_rows)
            axs[row, col].imshow(x0[i, :, :, :].permute(1, 2, 0).cpu().numpy(), cmap = cmap)
    
        for ax in axs.flat:
            ax.axis("off")

        xN_fig, axs = plt.subplots(self.num_rows, self.num_rows, figsize=(20, 20))
        for i in range(self.num_rows ** 2):
            row, col = divmod(i, self.num_rows)
            axs[row, col].imshow(xN[i, :, :, :].permute(1, 2, 0).cpu().numpy(), cmap = cmap)

        for ax in axs.flat:
            ax.axis("off")
        
        pl_module.logger.log_image(
            "Marginal distributions",
            [wandb.Image(x0_fig), wandb.Image(xN_fig)],
            caption = ["Samples from data distribution", "Samples from prior distribution"],
            step=trainer.global_step
            )
    
        plt.close("all")

class PlotImageSamplesCB(pl.Callback):
    def __init__(
        self, 
        num_rows : int = 5,
        ema_scope : bool = True
        ):
        super().__init__()
        self.num_rows = num_rows
        self.ema_scope = ema_scope

    def on_validation_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        iteration = pl_module.hparams.DSB_iteration
        device = pl_module.device

        is_backward = pl_module.hparams.training_backward
        title = "Backward" if is_backward else "Forward"
        if self.ema_scope:
            title += "_ema"
        original_dataset = trainer.datamodule.end_dataset_val if is_backward else trainer.datamodule.start_dataset_val
        original_xs = get_batch_from_dataset(original_dataset, self.num_rows ** 2, shuffle=True).to(device)
        sampled_xs = pl_module.sample(original_xs, forward = not is_backward, return_trajectory = False, clamp=True, ema_scope=self.ema_scope)

        original_xs = (original_xs + 1) / 2
        sampled_xs = (sampled_xs + 1) / 2

        original_xs = original_xs.permute(0, 2, 3, 1).cpu().detach().numpy()
        sampled_xs = sampled_xs.permute(0, 2, 3, 1).cpu().detach().numpy()

        fig = get_grid_fig(original_xs, sampled_xs, self.num_rows)
        pl_module.logger.log_image(f"iteration_{iteration}/{title} samples", [wandb.Image(fig)], step=trainer.global_step)
        plt.close("all")

class SanityCheckImagesCB(pl.Callback):
    def __init__(self):
        super().__init__()
    
    def on_train_start(self, trainer: pl.Trainer, pl_module : Union[FRDSB, TRDSB] ) -> None:
        self.xN = get_batch_from_dataset(trainer.datamodule.end_dataset_val, 5).to(pl_module.device)
        self.x0 = get_batch_from_dataset(trainer.datamodule.start_dataset_val, 5).to(pl_module.device)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: Union[FRDSB, TRDSB] ) -> None:
        """
        Checks if the model can reverse the process
        """
        pl_module.eval()
        iteration = pl_module.hparams.DSB_iteration
        is_backward = pl_module.hparams.training_backward
        title = "Backward" if is_backward else "Forward"
        
        data = self.x0 if is_backward else self.xN
        prior = pl_module.sample(data, forward = is_backward, return_trajectory=False, clamp=True, ema_scope=True)
        reconstruction = pl_module.sample(prior, forward = not is_backward, return_trajectory=False, clamp=True, ema_scope=True)
        
        data = (data + 1) / 2
        prior = (prior + 1) / 2
        reconstruction = (reconstruction + 1) / 2
        
        data = data.permute(0, 2, 3, 1).cpu().detach().numpy()
        prior = prior.permute(0, 2, 3, 1).cpu().detach().numpy()
        reconstruction = reconstruction.permute(0, 2, 3, 1).cpu().detach().numpy()
        
        fig, axs = plt.subplots(3, 5, figsize=(20, 12))
        for i in range(5):
            axs[0, i].imshow(data[i])
            axs[1, i].imshow(prior[i])
            axs[2, i].imshow(reconstruction[i])
            
        axs[0, 0].set_title("Data")
        axs[1, 0].set_title("Prior")
        axs[2, 0].set_title("Reconstruction")
        
        for ax in axs.flat:
            ax.axis("off")
            
        pl_module.logger.log_image(f"iteration_{iteration}/Sanity check {title}", [wandb.Image(fig)], step=trainer.global_step)
        plt.close("all")
    
class PlotImagesCB(pl.Callback):
    def __init__(self, ema_scope : bool = True):
        """
        Plots the forward and backward trajectory of the model on the validation set
        """
        super().__init__()
        self.ema_scope = ema_scope
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        device = pl_module.device
        x0 = get_batch_from_dataset(trainer.datamodule.start_dataset_val, 5).to(device)
        trajectory = pl_module.sample(x0, forward = True, return_trajectory = True, clamp=True, ema_scope=self.ema_scope).cpu()
        trajectory = (trajectory + 1) / 2
        
        fig = get_image_fig(trajectory)
        pl_module.logger.log_image("Initial forward trajectory", [wandb.Image(fig)])
        plt.close("all")
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        iteration = pl_module.hparams.DSB_iteration

        is_backward = pl_module.hparams.training_backward
        title = "Backward" if is_backward else "Forward"
        dataset = trainer.datamodule.end_dataset_val if is_backward else trainer.datamodule.start_dataset_val
        original_xs = get_batch_from_dataset(dataset, 5, shuffle=True).to(pl_module.device)

        trajectory = pl_module.sample(original_xs, forward = not is_backward, return_trajectory = True, clamp=True, ema_scope=self.ema_scope).cpu()
        trajectory = (trajectory + 1) / 2
        fig = get_image_fig(trajectory)
        pl_module.logger.log_image(f"iteration_{iteration}/{title} trajectory", [wandb.Image(fig)], step=trainer.global_step)
        plt.close("all")

        video_length_seconds = 5
        video_frames = trajectory.shape[0]
        video_fps = int(video_frames / video_length_seconds)
        
        if is_backward:
            trajectory = trajectory.flip(0)

        # save the first video
        video = trajectory[:, 0]
        if video.shape[1] == 1:
            video = video.repeat(1, 3, 1, 1)
        video = (video * 255).numpy().astype("uint8")
        pl_module.logger.log_video(f"iteration_{iteration}/{title} video", [video], step=trainer.global_step, fps=[video_fps])

class CalculateFID(pl.Callback):
    def __init__(self, num_samples : int = 1000, ema_scope : bool = True):
        super().__init__()
        self.num_samples = num_samples
        self.ema_scope = ema_scope

    def save_images_in_folder(self, images : torch.Tensor, folder : str):
        os.makedirs(folder, exist_ok=True)
        
        # if the folder doesnt contain num_samples images, delete it
        if len(os.listdir(folder)) <= self.num_samples:
            num_images = images.size(0)
            for i in range(num_images):
                img = images[i].permute(1, 2, 0).cpu().squeeze().numpy()
                img = (img + 1) / 2
                plt.imsave(f"{folder}/{i}.png", img)
        
    def on_train_start(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        wandb.define_metric("benchmarks/FID", step_metric="Iteration", summary="min")

        x0_dataset = trainer.datamodule.start_dataset
        xN_dataset = trainer.datamodule.end_dataset_val
        self.x0 = get_batch_from_dataset(x0_dataset, self.num_samples).to(pl_module.device)
        self.xN = get_batch_from_dataset(xN_dataset, self.num_samples).to(pl_module.device)
        # we will save xN as images in a folder to calculate FID later
        # the dataset have a unique identifier we will use as the folder name
        # if the folder doesnt already exist, loop over the images and save them
        identifier = x0_dataset.unique_identifier
        self.x0_folder = f"data/fid/{identifier}"
        self.save_images_in_folder(self.x0, self.x0_folder)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        iteration = pl_module.hparams.DSB_iteration
        if not pl_module.hparams.training_backward:
            pl_module.eval()
            x0_pred = pl_module.sample(self.xN, forward=False, return_trajectory=False, clamp=True, ema_scope=self.ema_scope)
            x0_pred = (x0_pred + 1) / 2
            random_folder_name = str(uuid.uuid4())
            generated_x0_folder = f"data/fid/{random_folder_name}"
            print("Calculating FID..")
            self.save_images_in_folder(x0_pred, generated_x0_folder)
            fid_value = fid.compute_fid(self.x0_folder, generated_x0_folder, num_workers=0, verbose=False)
            pl_module.logger.log_metrics({"benchmarks/FID": fid_value, "Iteration": iteration})

            # delete the generated images
            shutil.rmtree(generated_x0_folder)