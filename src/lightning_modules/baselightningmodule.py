import pytorch_lightning as pl

class BaseLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
    def _convert_dict_losses(self, losses, suffix = "", prefix = ""):
        if suffix:
            losses = {f"{k}/{suffix}": v for k, v in losses.items()}
        if prefix:
            losses = {f"{prefix}/{k}": v for k, v in losses.items()}
        return losses