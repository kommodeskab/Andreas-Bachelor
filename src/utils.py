import os
import hydra
from omegaconf import DictConfig
from datetime import datetime

def get_current_time() -> str:
    now = datetime.now()
    return now.strftime("%d%m%y%H%M%S")

def get_project_name_from_id(experiment_id : str) -> str | None:
    for dirpath, dirnames, _ in os.walk("logs"):
        if experiment_id in dirnames:
            return os.path.basename(dirpath)
        
    raise ValueError(f"Experiment {experiment_id} not found. Maybe it is on a different machine?")

def get_ckpt_path(experiment_id : str) -> str | None:
    if experiment_id is False:
        return None
    
    project_name = get_project_name_from_id(experiment_id)
    ckpt_folder = os.path.join(f"logs/{project_name}/{experiment_id}/checkpoints")
    ckpts = os.listdir(ckpt_folder)
    ckpts.sort()
    
    return os.path.join(ckpt_folder, ckpts[-1])

def instantiate_callbacks(callback_cfg : DictConfig | None) -> list:
    callbacks = []
    
    if callback_cfg is None:
        return callbacks
    
    for _, callback_params in callback_cfg.items():
        callback = hydra.utils.instantiate(callback_params)
        callbacks.append(callback)
        
    return callbacks