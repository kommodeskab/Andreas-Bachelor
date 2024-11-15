import os

model_dict = {
    "cat_32_epsilon": "091124235944",
    "cat_32_sample": "111124221939",
    "dog_32_sample": "121124165314",
    "dog_64_epsilon_pretrained": "131124165507",
    "cat_32_epsilon_pretrained": "131124224345",
    "dog_32_epsilon_pretrained": "131124224527",
    "cat_32_sample_pretrained": "141124231450",
    "dog_32_sample_pretrained": "141124193417",
}

def ckpt_path_from_id(model_id : str) -> str:
    folder_name = f"logs/diffusion/{model_dict[model_id]}/checkpoints"
    return os.path.join(folder_name, os.listdir(folder_name)[0])