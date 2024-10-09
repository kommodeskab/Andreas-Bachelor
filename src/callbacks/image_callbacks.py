from src.callbacks.plot_functions import get_grid_fig, get_image_fig
from src.lightning_modules.schrodinger_bridge import StandardDSB
import pytorch_lightning as pl
from src.callbacks.utils import get_batch_from_dataset
import wandb
import matplotlib.pyplot as plt
from cleanfid import fid
import os
import torch
import shutil
from pytorch_lightning.loggers import WandbLogger

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

        x0_fig.suptitle("Samples from data distribution", fontsize = 20)

        xN_fig, axs = plt.subplots(self.num_rows, self.num_rows, figsize=(20, 20))
        for i in range(self.num_rows ** 2):
            row, col = divmod(i, self.num_rows)
            axs[row, col].imshow(xN[i, :, :, :].permute(1, 2, 0).cpu().numpy(), cmap = cmap)

        for ax in axs.flat:
            ax.axis("off")

        xN_fig.suptitle("Samples from prior distribution", fontsize = 20)
        
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

        if pl_module.hparams.training_backward:
            xN = get_batch_from_dataset(trainer.datamodule.end_dataset_val, self.num_rows ** 2, shuffle=True).to(device)
            x0_pred = pl_module.sample(xN, forward = False, return_trajectory = False, clamp=True, ema_scope=True)
            xN = (xN + 1) / 2
            x0_pred = (x0_pred + 1) / 2
            xN = xN.permute(0, 2, 3, 1).cpu().detach().numpy()
            x0_pred = x0_pred.permute(0, 2, 3, 1).cpu().detach().numpy()

            fig = get_grid_fig(xN, x0_pred, self.num_rows)
            logger.log_image(f"iteration_{iteration}/Backward sampling", [wandb.Image(fig)], step=trainer.global_step)
        else:
            x0 = get_batch_from_dataset(trainer.datamodule.start_dataset_val, self.num_rows ** 2, shuffle=True).to(device)
            xN_pred = pl_module.sample(x0, forward = True, return_trajectory = False, clamp=True, ema_scope=True)
            x0 = (x0 + 1) / 2
            xN_pred = (xN_pred + 1) / 2
            x0 = x0.permute(0, 2, 3, 1).cpu().detach().numpy()
            xN_pred = xN_pred.permute(0, 2, 3, 1).cpu().detach().numpy()
            
            fig = get_grid_fig(x0, xN_pred, self.num_rows)
            logger.log_image(f"iteration_{iteration}/Forward sampling", [wandb.Image(fig)], step=trainer.global_step)
        
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
        trajectory = pl_module.sample(self.x0, forward = True, return_trajectory = True, clamp=True)
        trajectory = (trajectory + 1) / 2
        
        fig = get_image_fig(trajectory, "Initial forward trajectory")
        logger.log_image("Initial forward trajectory", [wandb.Image(fig)])
        plt.close("all")
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        iteration = pl_module.hparams.DSB_iteration
        logger : WandbLogger = trainer.logger

        if pl_module.hparams.training_backward:
            trajectory = pl_module.sample(self.xN, forward = False, return_trajectory = True, clamp=True, ema_scope=self.ema_scope)
            trajectory = (trajectory + 1) / 2

            fig = get_image_fig(trajectory, f"Backward trajectory (DSB-iteration: {iteration})")
            logger.log_image(f"iteration_{iteration}/Backward trajectory", [wandb.Image(fig)], step=trainer.global_step)
        else:
            trajectory = pl_module.sample(self.x0, forward = True, return_trajectory = True, clamp=True, ema_scope=self.ema_scope)
            trajectory = (trajectory + 1) / 2

            fig = get_image_fig(trajectory, f"Forward trajectory (DSB-iteration: {iteration})")
            logger.log_image(f"iteration_{iteration}/Forward trajectory", [wandb.Image(fig)], step=trainer.global_step)

        plt.close("all")

class TestInitialDiffusionCB(pl.Callback):
    def __init__(self, num_rows : int = 5):
        super().__init__()
        self.num_rows = num_rows
    
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
            x0_pred = pl_module.sample(noise, forward = False, clamp=True, ema_scope=True)
            x0_pred = (x0_pred + 1) / 2

            fig, axs = plt.subplots(self.num_rows, self.num_rows, figsize=(20, 20))
            cmap = "gray" if x0_pred.size(1) == 1 else None
            for i in range(self.num_rows ** 2):
                row, col = divmod(i, self.num_rows)
                axs[row, col].imshow(x0_pred[i, :, :, :].permute(1, 2, 0).cpu().numpy(), cmap = cmap)
                axs[row, col].axis("off")

            logger.log_image("Initial diffusion", [wandb.Image(fig)], step=trainer.global_step)
            plt.close("all")


class CalculateFID(pl.Callback):
    def __init__(self, num_samples : int = 1000):
        super().__init__()
        self.num_samples = num_samples

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
        if not pl_module.hparams.training_backward:
            pl_module.eval()
            x0_pred = pl_module.sample(self.xN, forward=False, return_trajectory=False, clamp=True, ema_scope=True)
            x0_pred = (x0_pred + 1) / 2
            generated_x0_folder = "data/fid/generated"
            print("Calculating FID..")
            self.save_images_in_folder(x0_pred, generated_x0_folder)
            fid_value = fid.compute_fid(self.x0_folder, generated_x0_folder, num_workers=0, verbose=False)
            logger.log_metrics({"benchmarks/FID": fid_value}, step=trainer.global_step)

            # delete the generated images
            shutil.rmtree(generated_x0_folder)