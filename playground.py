from src.lightning_modules import TRDSB
from src.networks import UNet2D
from src.lightning_modules.utils import ckpt_path_from_id
import matplotlib.pyplot as plt
import os
from src.dataset import EMNISTNoLabel
from src.callbacks.utils import get_batch_from_dataset
import copy

forward_model = UNet2D(
    in_channels=1,
    out_channels=1,
    block_out_channels=[32,32,64],
    down_block_types=["DownBlock2D","DownBlock2D","DownBlock2D"],
    dropout=0.1,
    layers_per_block=2,
    up_block_types=["UpBlock2D","UpBlock2D","UpBlock2D"],
    sample_size=[32,32],
)
backward_model = copy.deepcopy(forward_model)

folder_name = "logs/mnist_to_emnist/111124125830/checkpoints"
path = os.path.join(folder_name, os.listdir(folder_name)[0])
model = TRDSB.load_from_checkpoint(path, forward_model=forward_model, backward_model=backward_model, optimizer=None, scheduler=None, strict=False)

device = "cuda"

digits = EMNISTNoLabel(split="digits", img_size=16)
samples = get_batch_from_dataset(digits, 5).to(device)

letters = model.sample(samples, forward=True, return_trajectory=False, clamp=True)
reconstructions = model.sample(letters, forward=False, return_trajectory=False, clamp=True)

fig, axs = plt.subplots(3, 5)
for i, imgs in enumerate([samples, letters, reconstructions]):
    imgs = imgs.detach().cpu()
    imgs = (imgs + 1) / 2
    for j, img in enumerate(imgs):
        axs[i,j].imshow(img[0], cmap="gray")
        
plt.savefig("playground.png")