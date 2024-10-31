from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from src.utils import get_ckpt_path, instantiate_callbacks, get_current_time
import pytorch_lightning as pl
import os, hydra, torch

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["USE_FLASH_ATTENTION"] = "1"

@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg : DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed)

    cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    project_name, task_name = cfg.project_name, cfg.task_name
    
    print("Setting up logger..")
    logger = WandbLogger(
        **cfg.logger,
        project = project_name, 
        name = task_name, 
        version=get_current_time(), 
        )
    logger.experiment.config["cfg"] = cfg_yaml
    print(f"Config:\n\n{cfg_yaml}")
    
    print("Instantiating model and datamodule..")
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    print("Compiling model..")
    if cfg.compile:
        torch.compile(model)
    
    
    print("Instantiating callbacks..")
    callbacks = instantiate_callbacks(cfg.get("callbacks", None))

    print("Setting up trainer..")
    trainer = Trainer(
        **cfg.trainer, 
        logger = logger, 
        callbacks = callbacks
        )
        
    print("Beginning training..")
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    my_app()