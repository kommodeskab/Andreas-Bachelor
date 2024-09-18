from src.utils import get_ckpt_path
from clearml import Task
import pytorch_lightning as pl
from omegaconf import OmegaConf
import hydra

class PretrainedLightningModule(pl.LightningModule):
    def __init__(self, experiment_id : str):
        # get the config from clearml and instantiate the target_module
        task : Task = Task.get_task(task_id = experiment_id)
        cfg_str = task.get_configuration_object("OmegaConf")
        cfg = OmegaConf.create(cfg_str)
        target_module = hydra.utils.instantiate(cfg.model)
        
        # set this pretrained model as the target model
        self.__dict__.update(target_module.__dict__)
        self.__class__ = target_module.__class__
        
        # load the pretrained model's state dict
        ckpt_path = get_ckpt_path(experiment_id)
        self.load_from_checkpoint(ckpt_path)