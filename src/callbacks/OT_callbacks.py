from typing import Any
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from src.lightning_modules.schrodinger_bridge import StandardSchrodingerBridge
from src.callbacks.utils import get_batch_from_dataset

class SchrodingerChangeDataloaderCB(pl.Callback):
    def __init__(self):
        """
        Makes sure that the datamodule returns the correct data,
        i.e. that the training_backward attribute of the datamodule is the same as for the model.
        Handling whether the model is training backward or forward is determined inside the model itself.
        """
        super().__init__()
        
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert hasattr(trainer.datamodule.hparams, "training_backward"), "DataModule must have attribute training_backward"
        assert hasattr(pl_module.hparams, "training_backward"), "Model must have attribute training_backward"
        assert trainer.datamodule.hparams.training_backward == pl_module.hparams.training_backward, "DataModule and model must have the same training_backward"
        
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: StandardSchrodingerBridge) -> None:
        trainer.datamodule.hparams.training_backward = pl_module.hparams.training_backward

class PlotGammaScheduleCB(pl.Callback):
    def __init__(self):
        super().__init__()
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: StandardSchrodingerBridge) -> None:
        gammas = pl_module.gammas
        ts = torch.linspace(0, 1, len(gammas))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ts, gammas)
        ax.set_xlabel("t")
        ax.set_ylabel("gamma")
        ax.set_title("Gamma schedule")
        trainer.logger.experiment.add_figure("gammaschedule/Gamma schedule", fig, global_step=trainer.global_step)
        
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

        x0 = get_batch_from_dataset(start_dataset, self.num_samples).to(device)
        xT = get_batch_from_dataset(end_dataset, self.num_samples).to(device)

        if self.plot_trajectory:
            fig, ax = plt.subplots(2, 5, figsize=(20, 10))
            # first go forward
            trajectory = pl_module.sample(x0, forward = True, return_trajectory=True)
            traj_len = trajectory.shape[0]
            # only keep 5 trajectories, the start, the end and 3 in between
            traj_idx = [0, traj_len//4, traj_len//2, 3*traj_len//4, traj_len-1]
            trajectory = trajectory[traj_idx, :, :]
            min_x, max_x = trajectory[:, :, 0].min(), trajectory[:, :, 0].max()
            min_y, max_y = trajectory[:, :, 1].min(), trajectory[:, :, 1].max()

            for i in range(5):
                x, y = trajectory[i, :, 0].cpu().numpy(), trajectory[i, :, 1].cpu().numpy()
                ax[0, i].scatter(x, y, s = 1)
                ax[0, i].set_aspect('equal')
                ax[0, i].set_title(f"Step {traj_idx[i]}")
                ax[0, i].set_xlim(min_x, max_x)
                ax[0, i].set_ylim(min_y, max_y)

            # then go backward
            trajectory = pl_module.sample(xT, forward = False, return_trajectory=True)
            trajectory = trajectory[traj_idx, :, :]
            min_x, max_x = trajectory[:, :, 0].min(), trajectory[:, :, 0].max()
            min_y, max_y = trajectory[:, :, 1].min(), trajectory[:, :, 1].max()

            for i in range(5):
                x, y = trajectory[i, :, 0].cpu().numpy(), trajectory[i, :, 1].cpu().numpy()
                ax[1, i].scatter(x, y, s = 1)
                ax[1, i].set_aspect('equal')
                ax[1, i].set_xlim(min_x, max_x)
                ax[1, i].set_ylim(min_y, max_y)

            trainer.logger.experiment.add_figure("Forward and backward trajectory", fig, global_step=trainer.global_step)