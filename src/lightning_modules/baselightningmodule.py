import pytorch_lightning as pl

class BaseLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
    def _convert_dict_losses(self, losses, suffix = ""):
        return {f"{k}/{suffix}": v for k, v in losses.items()}