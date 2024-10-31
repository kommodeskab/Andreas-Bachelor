from src.callbacks.utils import get_batch_from_dataset
from src.lightning_modules.schrodinger_bridge import StandardDSB
import pytorch_lightning as pl
from src.callbacks.plot_functions import get_traj_fig
import wandb
import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal, kl_divergence

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
        fig = get_traj_fig(trajectory, num_points = self.num_points)
        pl_module.logger.log_image("Initial forward trajectory", [wandb.Image(fig)], caption=["Initial forward trajectory"])
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        iteration = pl_module.hparams.DSB_iteration

        is_backward = pl_module.hparams.training_backward
        title = "Backward" if is_backward else "Forward"
        original_xs = self.xN if is_backward else self.x0

        trajectory = pl_module.sample(original_xs, forward = not is_backward, return_trajectory=True).cpu()
        fig = get_traj_fig(trajectory, num_points = self.num_points)
        pl_module.logger.log_image(f"iteration_{iteration}/{title} trajectory", [wandb.Image(fig)], step=trainer.global_step)

        plt.close("all")
        
class GaussianTestCB(pl.Callback):
    def __init__(self, num_samples : int = 1000):
        super().__init__()
        self.num_samples = num_samples
        wandb.define_metric("benchmarks/KL-divergence", step_metric="Iteration", summary="min")
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        self.xN = get_batch_from_dataset(trainer.datamodule.end_dataset_val, self.num_samples).to(pl_module.device)
        
        real_mu, real_sigma = trainer.datamodule.start_dataset.mu, trainer.datamodule.start_dataset.sigma
        self.real_mu = torch.tensor(real_mu).to(pl_module.device)
        self.real_sigma = torch.tensor(real_sigma).to(pl_module.device)

        # calculate "baseline" KL divergence
        x0 = get_batch_from_dataset(trainer.datamodule.start_dataset_val, self.num_samples).to(pl_module.device)
        x0_mu, x0_sigma = x0.mean(dim = 0), x0.std(dim = 0)
        self.baseline_kl = self.calculate_kl_divergence(x0_mu, x0_sigma)

    def calculate_kl_divergence(self, pred_mu, pred_sigma):
        real_mu, real_sigma = self.real_mu, self.real_sigma
        dist1 = Normal(real_mu.expand_as(pred_mu), real_sigma.expand_as(pred_sigma))
        dist2 = Normal(pred_mu, pred_sigma)
        kl_div = kl_divergence(dist1, dist2).sum()
        return kl_div.item()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        iteration = pl_module.hparams.DSB_iteration

        if not pl_module.hparams.training_backward:
            x0_pred = pl_module.sample(self.xN, forward = False)
            pred_mu, pred_sigma = x0_pred.mean(dim = 0), x0_pred.std(dim = 0)
            # calculate the kl divergence assuming normal distributions
            kl_divergence = self.calculate_kl_divergence(pred_mu, pred_sigma)
            pl_module.logger.log_metrics({
                "benchmarks/kl": kl_divergence, 
                "benchmarks/baseline_kl": self.baseline_kl,
                "Iteration": iteration})