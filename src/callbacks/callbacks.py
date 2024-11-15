import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import matplotlib
from src.lightning_modules.schrodinger_bridge import StandardDSB
from src.callbacks.utils import get_batch_from_dataset, MMD
from src.callbacks.plot_functions import get_gamma_fig
import wandb

matplotlib.use('Agg')

class PlotGammaScheduleCB(pl.Callback):
    def __init__(self):
        """
        Callback to plot the gamma schedule.
        Plots the gamma schedule, gamma bar schedule, sigma backward schedule and sigma forward schedule.
        """
        super().__init__()
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        gammas = pl_module.gammas # first gamma is 0, and is never used (since indexing starts at 1)
        gammas_fig = get_gamma_fig(gammas, "Gamma")

        gammas_bar = pl_module.gammas_bar
        gammas_bar_fig = get_gamma_fig(gammas_bar, "Gamma bar")

        sigma_backward = pl_module.sigma_backward
        sigma_backward_fig = get_gamma_fig(sigma_backward, "Sigma backward")

        sigma_forward = pl_module.sigma_forward
        sigma_forward_fig = get_gamma_fig(sigma_forward, "Sigma forward")

        pl_module.logger.log_image(
            "gammaschedule",
            [wandb.Image(gammas_fig), wandb.Image(gammas_bar_fig), wandb.Image(sigma_backward_fig), wandb.Image(sigma_forward_fig)],
            caption = [f"Gamma schedule, T = {pl_module.hparams.T}", "Gamma bar schedule", "Sigma backward schedule", "Sigma forward schedule"],
            step=trainer.global_step
            )
                
        plt.close("all")

class MMDCB(pl.Callback):
    def __init__(self, num_samples : int = 1000):
        """
        Callback to calculate the MMD between the initial and final distributions.
        Also calculates the "baseline" MMD between two halves of the initial distribution.
        """
        super().__init__()
        self.num_samples = num_samples

    def on_train_start(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        wandb.define_metric("benchmarks/MMD", step_metric="Iteration", summary="min")
        
        self.xN = get_batch_from_dataset(trainer.datamodule.end_dataset_val, self.num_samples).to(pl_module.device)
        self.x0 = get_batch_from_dataset(trainer.datamodule.start_dataset_val, self.num_samples).to(pl_module.device)
        
        # calculate "baseline" MMD
        # split x0 into two halves
        x0_1, x0_2 = torch.split(self.x0, self.x0.shape[0] // 2)
        self.mmd_baseline = MMD(x0_1, x0_2, "rbf").item()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        iteration = pl_module.hparams.DSB_iteration
        if not pl_module.hparams.training_backward:
            x0_pred = pl_module.sample(self.xN, forward = False)
            mmd_value = MMD(x0_pred, self.x0, "rbf").item()
            pl_module.logger.log_metrics({
                "benchmarks/MMD": mmd_value, 
                "benchmarks/MMD_baseline": self.mmd_baseline,
                "Iteration": iteration})
