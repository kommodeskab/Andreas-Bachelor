from src.dataset import CelebADataset
import matplotlib.pyplot as plt

dataset = CelebADataset(attr = 20, on_or_off = True, img_size = 64)
print(len(dataset))
for i in range(10):
    img = dataset[i]
    img = (img + 1) / 2
    plt.imshow(img.permute(1, 2, 0))
    plt.savefig(f"afhq_{i}.png")