import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import matplotlib
from src.lightning_modules.schrodinger_bridge import StandardDSB
from src.callbacks.utils import get_batch_from_dataset, MMD
from src.callbacks.plot_functions import get_gamma_fig
from pytorch_lightning.loggers import WandbLogger
import wandb

matplotlib.use('Agg')

class PlotGammaScheduleCB(pl.Callback):
    def __init__(self):
        super().__init__()
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        logger : WandbLogger = trainer.logger

        gammas = pl_module.gammas[1:] # first gamma is 0, and is never used (since indexing starts at 1)
        gammas_fig = get_gamma_fig(gammas, "Gamma")

        gammas_bar = pl_module.gammas_bar
        gammas_bar_fig = get_gamma_fig(gammas_bar, "Gamma bar")

        sigma_backward = pl_module.sigma_backward[1:]
        sigma_backward_fig = get_gamma_fig(sigma_backward, "Sigma backward")

        sigma_forward = pl_module.sigma_forward[:-1]
        sigma_forward_fig = get_gamma_fig(sigma_forward, "Sigma forward")

        logger.log_image(
            "gammaschedule",
            [wandb.Image(gammas_fig), wandb.Image(gammas_bar_fig), wandb.Image(sigma_backward_fig), wandb.Image(sigma_forward_fig)],
            caption = ["Gamma schedule", "Gamma bar schedule", "Sigma backward schedule", "Sigma forward schedule"],
            step=trainer.global_step
            )
        
        plt.close("all")

class MMDCB(pl.Callback):
    def __init__(self, num_samples : int = 1000):
        super().__init__()
        self.num_samples = num_samples
        wandb.define_metric("benchmarks/MMD", step_metric="Iteration", summary="min")

    def on_train_start(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        self.xN = get_batch_from_dataset(trainer.datamodule.end_dataset_val, self.num_samples).to(pl_module.device)
        self.x0 = get_batch_from_dataset(trainer.datamodule.start_dataset_val, self.num_samples).to(pl_module.device)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: StandardDSB) -> None:
        pl_module.eval()
        logger : WandbLogger = trainer.logger
        iteration = pl_module.hparams.DSB_iteration
        if not pl_module.hparams.training_backward:
            x0_pred = pl_module.sample(self.xN, forward = False)
            mmd_value = MMD(x0_pred, self.x0, "rbf").item()
            logger.log_metrics({"benchmarks/MMD": mmd_value, "Iteration": iteration})
