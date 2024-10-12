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
from pytorch_lightning.loggers import WandbLogger
import wandb

class MarginalDistributionsImagesCB(pl.Callback):
    def __init__(self, num_rows : int = 5):
        super().__init__()
        self.num_rows = num_rows

    def on_train_start(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        logger : WandbLogger = trainer.logger

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
        
        logger.log_image(
            "Marginal distributions",
            [wandb.Image(x0_fig), wandb.Image(xN_fig)],
            caption = ["Samples from data distribution", "Samples from prior distribution"],
            step=trainer.global_step
            )
    
        plt.close("all")

class PlotImageSamplesCB(pl.Callback):
    def __init__(self, num_rows : int = 5):
        super().__init__()
        self.num_rows = num_rows

    def on_validation_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        iteration = pl_module.hparams.DSB_iteration
        device = pl_module.device
        logger : WandbLogger = trainer.logger

        is_backward = pl_module.hparams.training_backward
        title = "Backward" if is_backward else "Forward"
        original_dataset = trainer.datamodule.end_dataset_val if is_backward else trainer.datamodule.start_dataset_val
        original_xs = get_batch_from_dataset(original_dataset, self.num_rows ** 2, shuffle=True).to(device)
        sampled_xs = pl_module.sample(original_xs, forward = not is_backward, return_trajectory = False, clamp=True, ema_scope=True)

        original_xs = (original_xs + 1) / 2
        sampled_xs = (sampled_xs + 1) / 2

        original_xs = original_xs.permute(0, 2, 3, 1).cpu().detach().numpy()
        sampled_xs = sampled_xs.permute(0, 2, 3, 1).cpu().detach().numpy()

        fig = get_grid_fig(original_xs, sampled_xs, self.num_rows)
        logger.log_image(f"iteration_{iteration}/{title} samples", [wandb.Image(fig)], step=trainer.global_step)
        plt.close("all")

class SanityCheckImagesCB(pl.Callback):
    def __init__(self):
        super().__init__()
    
    def on_train_start(self, trainer: pl.Trainer, pl_module : Union[FRDSB, TRDSB] ) -> None:
        self.xN = get_batch_from_dataset(trainer.datamodule.end_dataset_val, 5).to(pl_module.device)
        self.x0 = get_batch_from_dataset(trainer.datamodule.start_dataset_val, 5).to(pl_module.device)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: Union[FRDSB, TRDSB] ) -> None:
        pl_module.eval()
        iteration = pl_module.hparams.DSB_iteration

        traj_len = pl_module.hparams.num_steps + 1
        traj_idx = [0, traj_len//4, traj_len//2, 3*traj_len//4, traj_len-1]

        fig, axs = plt.subplots(10, 4, figsize=(20, 40))

        training_backward = pl_module.hparams.training_backward
        original = self.x0 if training_backward else self.xN
        trajectory = pl_module.sample(original, forward=training_backward, return_trajectory=True, clamp=True, ema_scope=False)

        traj_idx = traj_idx[1:] if training_backward else traj_idx[:-1]
        trajectory = trajectory[traj_idx]
        cmap = "gray" if trajectory.size(2) == 1 else None

        for i, idx in enumerate(traj_idx):
            ks = pl_module.k_to_tensor(idx, 5)
            pred = pl_module.pred_x0(trajectory[i], ks) if training_backward else pl_module.pred_xN(trajectory[i], ks)
            pred = pred.clamp(-1, 1).permute(0, 2, 3, 1).cpu().numpy()
            pred = (pred + 1) / 2
            trajectory_copy = trajectory[i].permute(0, 2, 3, 1).cpu().numpy()

            for j in range(4):
                axs[2 * j, i].imshow(trajectory_copy[j], cmap=cmap)
                axs[2 * j + 1, i].imshow(pred[j], cmap=cmap)

        for ax in axs.flat:
            ax.axis("off")

        plt.tight_layout()
        logger : WandbLogger = trainer.logger
        title = "Backward" if training_backward else "Forward"
        logger.log_image(f"iteration_{iteration}/{title} sanity check", [wandb.Image(fig)], step=trainer.global_step)
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
        logger : WandbLogger = trainer.logger
        self.x0 = get_batch_from_dataset(trainer.datamodule.start_dataset_val, 5).to(device)
        self.xN = get_batch_from_dataset(trainer.datamodule.end_dataset_val, 5).to(device)
        trajectory = pl_module.sample(self.x0, forward = True, return_trajectory = True, clamp=True, ema_scope=self.ema_scope).cpu()
        trajectory = (trajectory + 1) / 2
        
        fig = get_image_fig(trajectory)
        logger.log_image("Initial forward trajectory", [wandb.Image(fig)])
        plt.close("all")
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        iteration = pl_module.hparams.DSB_iteration
        logger : WandbLogger = trainer.logger

        is_backward = pl_module.hparams.training_backward
        title = "Backward" if is_backward else "Forward"
        original_xs = self.xN if is_backward else self.x0

        trajectory = pl_module.sample(original_xs, forward = not is_backward, return_trajectory = True, clamp=True, ema_scope=self.ema_scope).cpu()
        trajectory = (trajectory + 1) / 2
        fig = get_image_fig(trajectory)
        logger.log_image(f"iteration_{iteration}/{title} trajectory", [wandb.Image(fig)], step=trainer.global_step)
        plt.close("all")

        video_length_seconds = 5
        video_frames = trajectory.shape[0]
        video_fps = int(video_frames / video_length_seconds)
        
        if is_backward:
            trajectory = trajectory.flip(0)

        for i in range(min(5, trajectory.size(1))):
            video = trajectory[:, i]
            if video.shape[1] == 1:
                video = video.repeat(1, 3, 1, 1)
            video = (video * 255).numpy().astype("uint8")
            logger.log_video(f"Videos", [video], step=trainer.global_step, caption=[f"Trajectory {i}"], fps=[video_fps])

class TestInitialDiffusionCB(pl.Callback):
    def __init__(self, num_rows : int = 5, ema_scope : bool = True):
        super().__init__()
        self.num_rows = num_rows
        self.ema_scope = ema_scope
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        hparams = pl_module.hparams
        if hparams.DSB_iteration == 1 and hparams.training_backward:
            pl_module.eval()
            logger : WandbLogger = trainer.logger
            device = pl_module.device

            # we need to find out the shape of the images
            # we will only use the images for finding the shape
            images = get_batch_from_dataset(trainer.datamodule.start_dataset_val, self.num_rows ** 2, shuffle=True)
            noise = torch.randn_like(images).to(device)
            x0_pred = pl_module.sample(noise, forward = False, clamp=True, ema_scope=self.ema_scope)
            x0_pred = (x0_pred + 1) / 2

            fig, axs = plt.subplots(self.num_rows, self.num_rows, figsize=(20, 20))
            cmap = "gray" if x0_pred.size(1) == 1 else None
            for i in range(self.num_rows ** 2):
                row, col = divmod(i, self.num_rows)
                axs[row, col].imshow(x0_pred[i, :, :, :].permute(1, 2, 0).cpu().numpy(), cmap = cmap)

            for ax in axs.flat:
                ax.axis("off")

            logger.log_image("Initial diffusion", [wandb.Image(fig)], step=trainer.global_step)
            plt.close("all")


class CalculateFID(pl.Callback):
    def __init__(self, num_samples : int = 1000, ema_scope : bool = True):
        super().__init__()
        self.num_samples = num_samples
        self.ema_scope = ema_scope
        wandb.define_metric("benchmarks/FID", step_metric="Iteration", summary="min")

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
        logger : WandbLogger = trainer.logger
        iteration = pl_module.hparams.DSB_iteration
        if not pl_module.hparams.training_backward:
            pl_module.eval()
            x0_pred = pl_module.sample(self.xN, forward=False, return_trajectory=False, clamp=True, ema_scope=self.ema_scope)
            x0_pred = (x0_pred + 1) / 2
            generated_x0_folder = "data/fid/generated"
            print("Calculating FID..")
            self.save_images_in_folder(x0_pred, generated_x0_folder)
            fid_value = fid.compute_fid(self.x0_folder, generated_x0_folder, num_workers=0, verbose=False)
            logger.log_metrics({"benchmarks/FID": fid_value, "Iteration": iteration})

            # delete the generated images
            shutil.rmtree(generated_x0_folder)