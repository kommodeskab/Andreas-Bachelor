from typing import Any
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from src.lightning_modules.schrodinger_bridge import StandardSchrodingerBridge

def has_converged(losses : list, patience : int) -> bool:
    if len(losses) < patience:
        return False
    
    min_loss = min(losses[:-patience])
    return all([l > min_loss for l in losses[-patience:]])

class SchrodingerChangeDataloaderCB(pl.Callback):
    def __init__(
        self,
        patience : int = 100,
        ):
        super().__init__()
        self.patience = patience
        
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert hasattr(trainer.datamodule.hparams, "training_backward"), "DataModule must have attribute training_backward"
        assert hasattr(pl_module.hparams, "training_backward"), "Model must have attribute training_backward"
        assert trainer.datamodule.hparams.training_backward == pl_module.hparams.training_backward, "DataModule and model must have the same training_backward"
        
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: StandardSchrodingerBridge) -> None:
        losses = pl_module.losses
        
        if has_converged(losses, self.patience):
            pl_module.hparams.training_backward = not pl_module.hparams.training_backward
            pl_module.losses = []
            trainer.datamodule.hparams.training_backward = pl_module.hparams.training_backward
        
class SchrodingerPlot2dCB(pl.Callback):
    def __init__(
        self, 
        num_samples : int = 10,
        plot_trajectory : bool = False,
        ):
        super().__init__()
        self.num_samples = num_samples
        self.plot_trajectory = plot_trajectory
        
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: StandardSchrodingerBridge) -> None:
        device = pl_module.device
        pl_module.eval()
        
        start_dataset = trainer.datamodule.start_dataset_train
        end_dataset = trainer.datamodule.end_dataset_train
        
        fig, ax = plt.subplots()
        x0 : torch.Tensor = start_dataset[:self.num_samples].to(device)
        if self.plot_trajectory:
            trajectory = pl_module.sample(x0, forward = True, return_trajectory = True)
            for i in range(trajectory.size(1)):
                # ax.plot(trajectory[:, i, 0].cpu().numpy(), trajectory[:, i, 1].cpu().numpy(), "o-", color = "blue", alpha = 0.5)
                x, y = trajectory[:, i, 0].cpu().numpy(), trajectory[:, i, 1].cpu().numpy()
                ax.scatter(x, y, c = range(len(x)), cmap = "coolwarm", s = 1)
        else: 
            ax.scatter(x0[:, 0].cpu().numpy(), x0[:, 1].cpu().numpy(), label = "Start (from dataset)", color = "red", alpha = 0.5, s = 1)
            xT = pl_module.sample(x0, forward = True, return_trajectory = False)
            ax.scatter(xT[:, 0].cpu().numpy(), xT[:, 1].cpu().numpy(), label = "End (generated)", color = "blue", alpha = 0.5, s = 1)
            ax.legend()
        ax.set_aspect('equal')
        ax.set_title("Forward process")
        title = "Forward process (with trajectory)" if self.plot_trajectory else "Forward process"
        trainer.logger.experiment.add_figure(title, fig, global_step=trainer.global_step)
                        
        fig, ax = plt.subplots()
        xT : torch.Tensor = end_dataset[:self.num_samples].to(device)
        if self.plot_trajectory:
            trajectory = pl_module.sample(xT, forward = False, return_trajectory = True)
            for i in range(trajectory.size(1)):
                # ax.plot(trajectory[:, i, 0].cpu().numpy(), trajectory[:, i, 1].cpu().numpy(), "o-", color = "red", alpha = 0.5)
                x, y = trajectory[:, i, 0].cpu().numpy(), trajectory[:, i, 1].cpu().numpy()
                ax.scatter(x, y, c = range(len(x)), cmap = "coolwarm", s = 1)
        else:
            ax.scatter(xT[:, 0].cpu().numpy(), xT[:, 1].cpu().numpy(), label = "End (from dataset)", color = "blue", alpha = 0.5, s = 1)
            x0 = pl_module.sample(xT, forward = False, return_trajectory = False)
            ax.scatter(x0[:, 0].cpu().numpy(), x0[:, 1].cpu().numpy(), label = "Start (generated)", color = "red", alpha = 0.5, s = 1)
            ax.legend()
        ax.set_aspect('equal')
        ax.set_title("Backward process")
        title = "Backward process (with trajectory)" if self.plot_trajectory else "Backward process"
        trainer.logger.experiment.add_figure(title, fig, global_step=trainer.global_step)
        
        pl_module.train()