import torch
import matplotlib.pyplot as plt

num_rows = 4
x0 = torch.randn(num_rows ** 2, 1, 16, 16)
xN_pred = torch.randn_like(x0)

x0 = x0.permute(0, 2, 3, 1).cpu().detach().numpy()
xN_pred = xN_pred.permute(0, 2, 3, 1).cpu().detach().numpy()

fig, axs = plt.subplots(num_rows, 2 * num_rows + 1, figsize=(16, 8))

for i in range(num_rows ** 2):
    row, col = i // num_rows, i % num_rows
    ax = axs[row, col]
    ax.imshow(x0[i])

    row, col = i // num_rows, (i % num_rows) + num_rows + 1
    print(row, col)
    ax = axs[row, col]
    ax.imshow(xN_pred[i])

for ax in axs.flatten():
    ax.axis("off")

axs[0, 0].set_title("Original", fontsize=16)
axs[0, num_rows + 1].set_title("Generated", fontsize=16)

plt.tight_layout()
plt.show()