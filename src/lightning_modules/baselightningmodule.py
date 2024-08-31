import pytorch_lightning as pl

class BaseLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
    def _convert_dict_losses(self, losses, suffix = "", prefix = ""):
        if suffix:
            return {f"{k}/{suffix}": v for k, v in losses.items()}
        elif prefix:
            return {f"{prefix}/{k}": v for k, v in losses.items()}
    
    def on_train_start(self) -> None:
        hparams = self._convert_dict_losses(self.hparams, prefix="hparams")
        self.log_dict(hparams)