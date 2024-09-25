from typing import Any, Dict
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import matplotlib
from src.lightning_modules.schrodinger_bridge import StandardDSB
from src.lightning_modules.reparameterized_dsb import TRDSB
from src.callbacks.utils import get_batch_from_dataset
import random

matplotlib.use('Agg')

class PlotGammaScheduleCB(pl.Callback):
    def __init__(self):
        super().__init__()
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        def make_gamma_plot(gammas, ylabel, title):
            ts = range(1, len(gammas)+1)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(ts, gammas)
            ax.set_xlabel("k")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            return fig
        
        gammas = pl_module.gammas[1:] # first gamma is 0, and is never used (since indexing starts at 1)
        fig = make_gamma_plot(gammas, "Gamma", "Gamma schedule")
        trainer.logger.experiment.add_figure("gammaschedule/Gamma schedule", fig, global_step=trainer.global_step)
        plt.close(fig)
        
        if hasattr(pl_module, "gammas_bar"):
            gammas_bar = pl_module.gammas_bar[1:]
            fig = make_gamma_plot(gammas_bar, "Gamma bar", "Gamma bar schedule")
            trainer.logger.experiment.add_figure("gammaschedule/Gamma bar schedule", fig, global_step=trainer.global_step)
            plt.close(fig)
            
        if hasattr(pl_module, "sigma_backward"):
            sigma_backward = pl_module.sigma_backward[1:]
            fig = make_gamma_plot(sigma_backward, "Sigma backward", "Sigma backward schedule")
            trainer.logger.experiment.add_figure("gammaschedule/Sigma backward schedule", fig, global_step=trainer.global_step)
            plt.close(fig)
            
        if hasattr(pl_module, "sigma_forward"):
            sigma_forward = pl_module.sigma_forward[1:]
            fig = make_gamma_plot(sigma_forward, "Sigma forward", "Sigma forward schedule")
            trainer.logger.experiment.add_figure("gammaschedule/Sigma forward schedule", fig, global_step=trainer.global_step)
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
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        device = pl_module.device
        iteration = pl_module.DSB_iteration

        # yes yes we should be using the validation set bla bla but it is too small with the current implementation
        x0 = get_batch_from_dataset(trainer.datamodule.start_dataset_train, self.num_points).to(device)
        xN = get_batch_from_dataset(trainer.datamodule.end_dataset_train, self.num_points).to(device)
        
        def get_traj_fig(trajectory, title):
            random_points = random.sample(range(self.num_points), 5)
            traj_len = trajectory.shape[0]
            traj_idx = [0, traj_len//4, traj_len//2, 3*traj_len//4, traj_len-1]
            trajectory_to_plot = trajectory[traj_idx, :, :]
            delta = 0.1
            min_x, max_x = trajectory[:, :, 0].min(), trajectory[:, :, 0].max()
            min_y, max_y = trajectory[:, :, 1].min(), trajectory[:, :, 1].max()
            min_x, max_x = min_x - delta, max_x + delta
            min_y, max_y = min_y - delta, max_y + delta

            fig, ax = plt.subplots(1, 5, figsize=(20, 4))
            colors = torch.sqrt(trajectory[0, :, 0] ** 2 + trajectory[0, :, 1] ** 2).tolist()
            for i in range(5):
                x, y = trajectory_to_plot[i, :, 0].cpu(), trajectory_to_plot[i, :, 1].cpu()
                ax[i].scatter(x, y, s = 1, c = colors, cmap = "viridis")
                
                x, y = trajectory_to_plot[i, random_points, 0].cpu(), trajectory_to_plot[i, random_points, 1].cpu()
                for j in range(len(random_points)):
                    ax[i].text(x[j] + 0.1, y[j] + 0.1, j + 1, fontsize=10, color = "black", bbox=dict(facecolor='white', alpha=0.9))
                ax[i].scatter(x, y, s = 20, c = "black")
                
                ax[i].set_aspect('equal')
                ax[i].set_title(f"Step {traj_idx[i]}")
                ax[i].set_xlim(min_x, max_x)
                ax[i].set_ylim(min_y, max_y)
            
            fig.suptitle(title, fontsize = 20)
            
            return fig
            
        trajectory = pl_module.sample(x0, forward = True, return_trajectory=True).cpu()
        fig = get_traj_fig(trajectory, f"Forward trajectory (Iteration: {iteration})")
        trainer.logger.experiment.add_figure(f"iteration_{iteration}/Forward trajectory", fig, global_step=trainer.global_step)
        plt.close(fig)
        
        trajectory = pl_module.sample(xN, forward = False, return_trajectory=True).cpu()
        fig = get_traj_fig(trajectory, f"Backward trajectory (Iteration: {iteration})")
        trainer.logger.experiment.add_figure(f"iteration_{iteration}/Backward trajectory", fig, global_step=trainer.global_step)
        plt.close(fig)
        
class PlotImagesCB(pl.Callback):
    def __init__(self):
        """
        Plots the forward and backward trajectory of the model on the validation set
        """
        super().__init__()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        device = pl_module.device
        iteration = pl_module.DSB_iteration

        x0 = get_batch_from_dataset(trainer.datamodule.start_dataset_val, 5).to(device)
        trajectory = pl_module.sample(x0, forward = True, return_trajectory = True, clamp=True)
        # the output is between -1  and 1. We need to scale it to 0-1 for plotting
        trajectory = (trajectory + 1) / 2

        traj_len = trajectory.shape[0]
        traj_idx = [0, traj_len//4, traj_len//2, 3*traj_len//4, traj_len-1]

        fig, ax = plt.subplots(5, 5, figsize=(20, 20))
        for i in range(5): # i is the index of the trajectory, which should be along the x-axis (columns)
            for j in range(5): # j is the index of the image, which should be along the y-axis (rows)
                img = trajectory[traj_idx[i], j, :, :, :].permute(1, 2, 0).cpu()
                ax[i, j].imshow(img)
                ax[i, j].axis("off")
                if j == 2:
                    ax[i, j].set_title(f"Step {traj_idx[i]}", fontsize = 20)

        fig.suptitle(f"Forward trajectory (DSB-iteration: {iteration})", fontsize = 40)
        trainer.logger.experiment.add_figure(f"iteration_{iteration}/Forward trajectory", fig, global_step=trainer.global_step)
        plt.close(fig)
        
        xN = get_batch_from_dataset(trainer.datamodule.end_dataset_val, 5).to(device)
        trajectory = pl_module.sample(xN, forward = False, return_trajectory = True)
        trajectory = (trajectory + 1) / 2

        fig, ax = plt.subplots(5, 5, figsize=(20, 20))
        for i in range(5):
            for j in range(5):
                img = trajectory[traj_idx[i], j, :, :, :].permute(1, 2, 0).cpu()
                ax[i, j].imshow(img)
                ax[i, j].axis("off")
                if j == 2:
                    ax[i, j].set_title(f"Step {traj_idx[i]}", fontsize = 20)

        fig.suptitle(f"Backward trajectory (DSB-iteration: {iteration})", fontsize = 40)
        trainer.logger.experiment.add_figure(f"iteration_{iteration}/Backward trajectory", fig, global_step=trainer.global_step)
        plt.close(fig)
        
class DebugImagesCB(pl.Callback):
    def __init__(self):
        super().__init__()
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        x0 = get_batch_from_dataset(trainer.datamodule.start_dataset_val, 1)
        xN = get_batch_from_dataset(trainer.datamodule.end_dataset_val, 1)
        
        if isinstance(pl_module, TRDSB):
            forward_trajectory = pl_module.sample(x0, forward = True, return_trajectory = True)
            backward_trajectory = pl_module.sample(xN, forward = False, return_trajectory = True)
            
            traj_len = forward_trajectory.shape[0]
            traj_idx = [0, traj_len//4, traj_len//2, 3*traj_len//4, traj_len-1]
            
            forward_trajectory = forward_trajectory[traj_idx, :, :, :, :]
            backward_trajectory = backward_trajectory[traj_idx, :, :, :, :]
            
            fig, ax = plt.subplots(5, 2, figsize=(20, 8))
            for i in range(5):
                xk = forward_trajectory[i, 0, :, :, :]
                k = traj_idx[i]
                ks = pl_module.k_to_tensor(k, 1)
                x0_pred = pl_module.backward_call(xk, ks)
                
                ax[i, 0].imshow(xk.squeeze().cpu().permute(1, 2, 0))
                ax[i, 0].axis("off")
                ax[i, 0].set_title(f"Forward step {k}")
                
                ax[i, 1].imshow(x0_pred.squeeze().cpu().permute(1, 2, 0))
                ax[i, 1].axis("off")
                ax[i, 1].set_title(f"$x_0$ prediction")
                
            fig.suptitle("Forward trajectory", fontsize = 20)
            trainer.logger.experiment.add_figure("debug/Forward trajectory", fig, global_step=trainer.global_step)
            plt.close(fig)
            
            fig, ax = plt.subplots(5, 2, figsize=(20, 8))
            for i in range(5):
                xk = backward_trajectory[i, 0, :, :, :]
                k = traj_idx[i]
                ks = pl_module.k_to_tensor(k, 1)
                xN_pred = pl_module.forward_call(xk, ks)
                
                ax[i, 0].imshow(xk.squeeze().cpu().permute(1, 2, 0))
                ax[i, 0].axis("off")
                ax[i, 0].set_title(f"Backward step {k}")
                
                ax[i, 1].imshow(xN_pred.squeeze().cpu().permute(1, 2, 0))
                ax[i, 1].axis("off")
                ax[i, 1].set_title(f"$x_N$ prediction")
                
            fig.suptitle("Backward trajectory", fontsize = 20)
            trainer.logger.experiment.add_figure("debug/Backward trajectory", fig, global_step=trainer.global_step)
            plt.close(fig)
                
class GaussianTestCB(pl.Callback):
    def __init__(
            self,
            num_samples : int = 1000,
            ):
        super().__init__()
        self.num_samples = num_samples
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.gaussian_test_results = {}

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        current_dsb_iteration = pl_module.DSB_iteration

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

            trainer.logger.experiment.add_figure("Mu and sigma error", fig, global_step=trainer.global_step)
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
