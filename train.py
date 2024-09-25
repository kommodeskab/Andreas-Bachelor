from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from src.utils import get_ckpt_path, instantiate_callbacks
import pytorch_lightning as pl
import os, hydra, torch
import wandb

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["USE_FLASH_ATTENTION"] = "1"

@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg : DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed)

    cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    print(f"Config:\n\n{cfg_yaml}")
    project_name, task_name = cfg.project_name, cfg.task_name
    wandb.init()

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    if cfg.compile:
        torch.compile(model)
    
    logger = WandbLogger(project = project_name, name = task_name, **cfg.logger)
    
    callbacks = instantiate_callbacks(cfg.get("callbacks", None))
    
    trainer = Trainer(**cfg.trainer, logger = logger, callbacks = callbacks)
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    my_app()