from .baselightningmodule import BaseLightningModule
from diffusers import DDPMScheduler
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

class DDPM(BaseLightningModule):
    def __init__(
        self,
        model : torch.nn.Module,
        num_train_timesteps : int,
        optimizer : Optimizer,
        prediction_type: str = 'sample',
        beta_start : float =  0.0001,
        beta_end : float = 0.02,
        lr_patience : int = 20,
        inference_steps : int | None = None,
    ):
        """
        A PyTorch Lightning module for training a Diffusion Probabilistic Model (DDPM).
        The model is trained using a DDPMScheduler, which adds noise to the input data.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "optimizer"])
        self.model = model
        self.scheduler = DDPMScheduler(
            num_train_timesteps = num_train_timesteps,
            beta_start = beta_start,
            beta_end = beta_end,
            prediction_type = prediction_type,
        )
        self.mse = torch.nn.MSELoss()
        self.partial_opt = optimizer
        inference_steps = inference_steps or num_train_timesteps
        self.set_timesteps(inference_steps)
    
    def set_timesteps(self, num_inference_timesteps : int) -> None:
        self.scheduler.set_timesteps(num_inference_timesteps)
        
    def forward(self, x : torch.Tensor, timesteps : torch.Tensor):
        return self.model(x, timesteps)
    
    def sample_timesteps(self, batch_size : int):
        return torch.randint(0, self.hparams.num_train_timesteps, (batch_size,)).to(self.device)
    
    def t_to_tensor(self, t : int, batch_size : int):
        return torch.full((batch_size,), t, dtype=torch.float32).to(self.device)
    
    def _common_step(self, batch):
        timesteps = self.sample_timesteps(batch.size(0))
        eps = torch.randn_like(batch)
        noised_batch = self.scheduler.add_noise(batch, eps, timesteps)
        model_output = self(noised_batch, timesteps)
        target = batch if self.hparams.prediction_type == 'sample' else eps
        loss = self.mse(model_output, target)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    @torch.no_grad()
    def sample(self, noise : torch.Tensor, return_trajectory : bool = False, clamp : bool = False):
        self.model.eval()
        batch_size = noise.size(0)
        sample = noise
        trajectory = [sample]
        for t in self.scheduler.timesteps:
            timestep = self.t_to_tensor(t, batch_size)
            model_output = self.model(sample, timestep)
            sample = self.scheduler.step(model_output, t, sample).prev_sample
            trajectory.append(sample)
            
        trajectory = torch.stack(trajectory, dim=0)
        if clamp:
            trajectory = trajectory.clamp(-1, 1)
            
        if return_trajectory:
            return trajectory
        
        return trajectory[-1]
    
    def configure_optimizers(self):
        optim = self.partial_opt(self.model.parameters())
        sch = ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=self.hparams.lr_patience, verbose=True)
        return {
            "optimizer" : optim,
            "lr_scheduler" : sch,
            "monitor" : "val_loss",
        }