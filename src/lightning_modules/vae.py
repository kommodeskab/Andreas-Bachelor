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
        z = self._sample_z(mu, log_var)
        x_hat = self.decoder(z)
    
        loss, kl_divergence, reconstruction_loss = self._calculate_loss(x, x_hat, mu, log_var)

        return {
            "kl_divergence": kl_divergence,
            "reconstruction_loss": reconstruction_loss,
            "loss": loss,
        }

    def _sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def _calculate_loss(self, x, x_hat, mu, log_var):
        reconstruction_loss = self.mse(x_hat, x)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = reconstruction_loss + self.beta * kl_divergence
        return loss, kl_divergence, reconstruction_loss
    
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
    