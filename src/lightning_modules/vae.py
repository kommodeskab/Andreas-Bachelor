import torch
from src.lightning_modules.baselightningmodule import BaseLightningModule

class SimpleVAE(BaseLightningModule):
    def __init__(
        self,
        encoder : torch.nn.Module,
        decoder : torch.nn.Module,
        beta : float = 1.0,
        ):
        super().__init__()
        self.save_hyperparameters(ignore=("encoder", "decoder"))
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.mse = torch.nn.MSELoss()
        
    def common_step(self, batch):
        x, _ = batch
        mu, log_var = self.encoder(x)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        x_hat = self.decoder(z)
        reconstruction_loss = self.mse(x_hat, x)
        loss = reconstruction_loss + self.beta * kl_divergence
        return {
            "kl_divergence": kl_divergence,
            "reconstruction_loss": reconstruction_loss,
            "loss": loss,
        }
    
    def training_step(self, batch, batch_idx):
        losses = self.common_step(batch)
        self.log_dict(self._convert_dict_losses (losses, suffix="train"), prog_bar=True)
        return losses["loss"]
    
    def validation_step(self, batch, batch_idx):
        losses = self.common_step(batch)
        self.log_dict(self._convert_dict_losses(losses, suffix="val"), prog_bar=True)
        return losses["loss"]
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 1e-3)
    