import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import matplotlib
from src.lightning_modules.schrodinger_bridge import StandardDSB
from src.callbacks.utils import get_batch_from_dataset
import wandb
from src.callbacks.plot_functions import get_gamma_fig, get_traj_fig, get_image_fig

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
        wandb.log({"Initial forward trajectory": wandb.Image(fig)}, step=trainer.global_step)
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
        trajectory = pl_module.sample(self.x0, forward = True, return_trajectory = True, clamp=True)
        trajectory = (trajectory + 1) / 2
        
        fig = get_image_fig(trajectory, "Initial forward trajectory")
        wandb.log({"Initial forward trajectory": wandb.Image(fig)}, step=trainer.global_step)
        plt.close(fig)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        iteration = pl_module.hparams.DSB_iteration

        trajectory = pl_module.sample(self.x0, forward = True, return_trajectory = True, clamp=True)
        trajectory = (trajectory + 1) / 2

        fig = get_image_fig(trajectory, f"Forward trajectory (DSB-iteration: {iteration})")
        wandb.log({f"iteration_{iteration}/Forward trajectory": wandb.Image(fig)}, step=trainer.global_step)
        plt.close(fig)
        
        trajectory = pl_module.sample(self.xN, forward = False, return_trajectory = True)
        trajectory = (trajectory + 1) / 2

        fig = get_image_fig(trajectory, f"Backward trajectory (DSB-iteration: {iteration})")
        wandb.log({f"iteration_{iteration}/Backward trajectory": wandb.Image(fig)}, step=trainer.global_step)
        plt.close(fig)
                
class GaussianTestCB(pl.Callback):
    def __init__(self, num_samples : int = 1000):
        super().__init__()
        self.num_samples = num_samples
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.gaussian_test_results = {}

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        current_dsb_iteration = pl_module.hparams.DSB_iteration

        x0 = get_batch_from_dataset(trainer.datamodule.start_dataset_val, self.num_samples).to(pl_module.device)
        xN_pred = pl_module.sample(x0, forward = True)
        xN_pred_mu, xN_pred_sigma = xN_pred.mean(dim = 0), xN_pred.std(dim = 0)
        xN_real_mu, xN_real_sigma = trainer.datamodule.end_mu.to(pl_module.device), trainer.datamodule.end_sigma.to(pl_module.device)
        mu_error = torch.norm(xN_pred_mu - xN_real_mu).item()
        sigma_error = torch.norm(xN_pred_sigma - xN_real_sigma).item()
        
        pl_module.gaussian_test_results[current_dsb_iteration] = {
            "mu_error": mu_error,
            "sigma_error": sigma_error
        }

        if current_dsb_iteration > 1:
            mu_errors = [v["mu_error"] for v in pl_module.gaussian_test_results.values()]
            sigma_errors = [v["sigma_error"] for v in pl_module.gaussian_test_results.values()]

            plt.plot(mu_errors, label = "Mu error")
            plt.plot(sigma_errors, label = "Sigma error")
            plt.xlabel("Iteration")
            plt.ylabel("Error")
            plt.legend()
            plt.title("Mu and sigma error")
            fig = plt.gcf()

            wandb.log({"Mu and sigma error": wandb.Image(fig)}, step=trainer.global_step)
            plt.close(fig)
            
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
