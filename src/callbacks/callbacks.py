import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import matplotlib
from src.lightning_modules.schrodinger_bridge import StandardDSB
from src.callbacks.utils import get_batch_from_dataset
import wandb
from src.callbacks.plot_functions import get_gamma_fig, get_traj_fig, get_image_fig, get_grid_fig

matplotlib.use('Agg')

class PlotGammaScheduleCB(pl.Callback):
    def __init__(self):
        super().__init__()
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        gammas = pl_module.gammas[1:] # first gamma is 0, and is never used (since indexing starts at 1)
        fig = get_gamma_fig(gammas, "Gamma", "Gamma schedule")
        wandb.log({"gammaschedule/Gamma schedule": wandb.Image(fig)}, step=trainer.global_step)
        plt.close(fig)
        
        if hasattr(pl_module, "gammas_bar"):
            gammas_bar = pl_module.gammas_bar[1:]
            fig = get_gamma_fig(gammas_bar, "Gamma bar", "Gamma bar schedule")
            wandb.log({"gammaschedule/Gamma bar schedule": wandb.Image(fig)}, step=trainer.global_step)
            plt.close(fig)
            
        if hasattr(pl_module, "sigma_backward"):
            sigma_backward = pl_module.sigma_backward[1:]
            fig = get_gamma_fig(sigma_backward, "Sigma backward", "Sigma backward schedule")
            wandb.log({"gammaschedule/Sigma backward schedule": wandb.Image(fig)}, step=trainer.global_step)
            plt.close(fig)
            
        if hasattr(pl_module, "sigma_forward"):
            sigma_forward = pl_module.sigma_forward[1:]
            fig = get_gamma_fig(sigma_forward, "Sigma forward", "Sigma forward schedule")
            wandb.log({"gammaschedule/Sigma forward schedule": wandb.Image(fig)}, step=trainer.global_step)
            plt.close(fig)
        
class Plot2dCB(pl.Callback):
    def __init__(
        self, 
        num_points : int = 1000,
        num_trajectories : int = 5,
        ):
        super().__init__()
        self.num_points = num_points
        self.num_trajectories = num_trajectories
        
    def on_train_start(self, trainer : pl.Trainer, pl_module : StandardDSB) -> None:
        pl_module.eval()
        self.x0 = get_batch_from_dataset(trainer.datamodule.start_dataset_train, self.num_points).to(pl_module.device)
        self.xN = get_batch_from_dataset(trainer.datamodule.end_dataset_train, self.num_points).to(pl_module.device)
        trajectory = pl_module.sample(self.x0, forward=True, return_trajectory=True).cpu()
        fig = get_traj_fig(trajectory, "Initial forward trajectory", num_points = self.num_points)
        wandb.log({"Initial forward trajectory/Initial forward trajectory": wandb.Image(fig)}, step=trainer.global_step)
        plt.close(fig)
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        iteration = pl_module.hparams.DSB_iteration
        
        trajectory = pl_module.sample(self.x0, forward = True, return_trajectory=True).cpu()
        fig = get_traj_fig(trajectory, f"Forward trajectory (Iteration: {iteration})", num_points = self.num_points)
        wandb.log({f"iteration_{iteration}/Forward trajectory": wandb.Image(fig)}, step=trainer.global_step)
        plt.close(fig)
        
        trajectory = pl_module.sample(self.xN, forward = False, return_trajectory=True).cpu()
        fig = get_traj_fig(trajectory, f"Backward trajectory (Iteration: {iteration})", num_points = self.num_points)
        wandb.log({f"iteration_{iteration}/Backward trajectory": wandb.Image(fig)}, step=trainer.global_step)
        plt.close(fig)
 
class PlotImagesCB(pl.Callback):
    def __init__(self):
        """
        Plots the forward and backward trajectory of the model on the validation set
        """
        super().__init__()
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        device = pl_module.device
        self.x0 = get_batch_from_dataset(trainer.datamodule.start_dataset_val, 5).to(device)
        self.xN = get_batch_from_dataset(trainer.datamodule.end_dataset_val, 5).to(device)
        trajectory = pl_module.sample(self.x0, forward = True, return_trajectory = True, clamp=True, ema_scope=True)
        trajectory = (trajectory + 1) / 2
        
        fig = get_image_fig(trajectory, "Initial forward trajectory")
        wandb.log({"Initial forward trajectory/Initial forward trajectory": wandb.Image(fig)}, step=trainer.global_step)
        plt.close(fig)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        iteration = pl_module.hparams.DSB_iteration

        trajectory = pl_module.sample(self.x0, forward = True, return_trajectory = True, clamp=True, ema_scope=True)
        trajectory = (trajectory + 1) / 2

        fig = get_image_fig(trajectory, f"Forward trajectory (DSB-iteration: {iteration})")
        wandb.log({f"iteration_{iteration}/Forward trajectory": wandb.Image(fig)}, step=trainer.global_step)
        plt.close(fig)

        trajectory = pl_module.sample(self.xN, forward = False, return_trajectory = True, clamp=True, ema_scope=True)
        trajectory = (trajectory + 1) / 2

        fig = get_image_fig(trajectory, f"Backward trajectory (DSB-iteration: {iteration})")
        wandb.log({f"iteration_{iteration}/Backward trajectory": wandb.Image(fig)}, step=trainer.global_step)
        plt.close(fig)

class PlotImageSamplesCB(pl.Callback):
    def __init__(self, num_rows : int = 5):
        super().__init__()
        self.num_rows = num_rows

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        iteration = pl_module.hparams.DSB_iteration
        device = pl_module.device

        xN = get_batch_from_dataset(trainer.datamodule.end_dataset_val, self.num_rows ** 2, shuffle=True).to(device)
        x0_pred = pl_module.sample(xN, forward = False, return_trajectory = False, clamp=True, ema_scope=True)
        xN = (xN + 1) / 2
        x0_pred = (x0_pred + 1) / 2
        xN = xN.permute(0, 2, 3, 1).cpu().detach().numpy()
        x0_pred = x0_pred.permute(0, 2, 3, 1).cpu().detach().numpy()

        fig = get_grid_fig(xN, x0_pred, self.num_rows)
        wandb.log({f"iteration_{iteration}/Backward sampling": wandb.Image(fig)}, step=trainer.global_step)
        plt.close(fig)
        
        x0 = get_batch_from_dataset(trainer.datamodule.start_dataset_val, self.num_rows ** 2, shuffle=True).to(device)
        xN_pred = pl_module.sample(x0, forward = True, return_trajectory = False, clamp=True, ema_scope=True)
        x0 = (x0 + 1) / 2
        xN_pred = (xN_pred + 1) / 2
        x0 = x0.permute(0, 2, 3, 1).cpu().detach().numpy()
        xN_pred = xN_pred.permute(0, 2, 3, 1).cpu().detach().numpy()
        
        fig = get_grid_fig(x0, xN_pred, self.num_rows)
        wandb.log({f"iteration_{iteration}/Forward sampling": wandb.Image(fig)}, step=trainer.global_step)
        plt.close(fig)

class GaussianTestCB(pl.Callback):
    def __init__(self, num_samples : int = 1000):
        super().__init__()
        self.num_samples = num_samples
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        self.kl_divergences = {}

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        iteration = pl_module.hparams.DSB_iteration
        pl_module.eval()

        x0 = get_batch_from_dataset(trainer.datamodule.start_dataset_val, self.num_samples).to(pl_module.device)
        xN_pred = pl_module.sample(x0, forward = True)
        pred_mu, pred_sigma = xN_pred.mean(dim = 0), xN_pred.std(dim = 0)
        real_mu, real_sigma = trainer.datamodule.end_dataset.mu, trainer.datamodule.end_dataset.sigma
        # calculate the kl divergence assuming normal distributions
        kl_divergence : torch.Tensor = 0.5 * (torch.log(real_sigma / pred_sigma) + (pred_sigma ** 2 + (pred_mu - real_mu) ** 2) / real_sigma - 1)
        kl_divergence = kl_divergence.sum().item()
        
        self.kl_divergences[iteration] = kl_divergence

        if iteration > 1:
            divergences = self.kl_divergences.values()

            plt.plot(range(1, iteration + 1), divergences, "o-")
            plt.grid(True)
            plt.xticks(range(1, iteration + 1))
            plt.xlabel("DSB iteration")
            plt.ylabel("KL divergence")
            plt.title("KL divergence between predicted and real distribution")
            wandb.log({f"iteration_{iteration}/KL divergence": wandb.Image(plt)}, step=trainer.global_step)
            plt.close()
            
class PlotAudioCB(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        device = pl_module.device
        sample_rate = trainer.datamodule.sample_rate

        x0 = get_batch_from_dataset(trainer.datamodule.start_dataset_val, 5).to(device)
        xN_pred = pl_module.sample(x0, forward = True)

        for i in range(5):
            start_audio, end_audio = x0[i, :, :].cpu(), xN_pred[i, :, :].cpu()
            concat_audio = torch.concatenate([start_audio, end_audio], dim = 0)
            trainer.logger.experiment.add_audio(f"Forward_sampling_{i}", concat_audio, global_step=trainer.global_step, sample_rate=sample_rate)

        xN = get_batch_from_dataset(trainer.datamodule.end_dataset_val, 5).to(device)
        x0_pred = pl_module.sample(xN, forward = False)

        for i in range(5):
            start_audio, end_audio = xN[i, :, :].cpu(), x0_pred[i, :, :].cpu()
            concat_audio = torch.concatenate([start_audio, end_audio], dim = 0)
            trainer.logger.experiment.add_audio(f"Backward_sampling_{i}", concat_audio, global_step=trainer.global_step, sample_rate=sample_rate)
