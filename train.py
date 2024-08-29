import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from clearml import Task
from omegaconf import OmegaConf
from src.utils import get_ckpt_path, instantiate_callbacks
import torch
import pytorch_lightning as pl

@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg : DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed)

    cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    print(f"Config:\n\n{cfg_yaml}")
    
    project_name, task_name, experiment_id = cfg.project_name, cfg.task_name, cfg.experiment_id
    
    task : Task = Task.init(project_name = project_name, task_name = task_name, continue_last_task=experiment_id)
    task.upload_artifact("cfg_yaml", cfg_yaml)
    
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    
    logger = TensorBoardLogger("logs", name = cfg.project_name, version = task.id, default_hp_metric = False)
    
    callbacks = instantiate_callbacks(cfg.get("callbacks", None))
    
    trainer = Trainer(**cfg.trainer, logger = logger, callbacks = callbacks)
    trainer.fit(model, datamodule, ckpt_path=get_ckpt_path(experiment_id))

if __name__ == "__main__":
    my_app()