from pytorch_lightning import Callback
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch

class VAECallback(Callback):
    def __init__(self, num_samples = 16):
        super().__init__()
        self.num_samples = num_samples
    
    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        num_samples = self.num_samples
        dataloader = DataLoader(trainer.datamodule.val_dataset, batch_size = num_samples)
        samples, _ = next(iter(dataloader))
        samples = samples.to(pl_module.device)
        
        with torch.no_grad():
            z = pl_module.encoder(samples)
            reconstructions = pl_module.decoder(z)
        
        fig, axs = plt.subplots(num_samples, 2, figsize = (10, 20))
        
        for i in range(num_samples):
            axs[i, 0].imshow(samples[i].cpu().numpy().squeeze(), cmap = 'gray')
            axs[i, 1].imshow(reconstructions[i].detach().cpu().numpy().squeeze(), cmap = 'gray')
            
        trainer.logger.experiment.add_figure('reconstructions', fig, global_step = trainer.global_step)
        
        pl_module.train()