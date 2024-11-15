import matplotlib.pyplot as plt
import random
import torch
import matplotlib

matplotlib.use('Agg')

def get_gamma_fig(gammas, ylabel):
    """
    Plot the gamma values as a simple line plot.
    """
    ts = range(1, len(gammas)+1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts, gammas)
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    return fig

def get_image_fig(trajectory : torch.Tensor):
    """
    Plot the trajectory as a 5x5 grid of images.
    The x-axis is the batch index, and the y-axis is the time index.
    """
    traj_len = trajectory.shape[0]
    traj_idx = [0, traj_len//4, traj_len//2, 3*traj_len//4, traj_len-1]
    cmap = "gray" if trajectory.size(2) == 1 else None

    # Plot the trajectory as a 5x5 grid of images
    fig, ax = plt.subplots(5, 5, figsize=(20, 20))
    for i in range(5): 
        for j in range(5): 
            img = trajectory[traj_idx[i], j, :, :, :].permute(1, 2, 0).numpy()
            ax[i, j].imshow(img, cmap = cmap)
            ax[i, j].axis("off")
            if j == 2:
                ax[i, j].set_title(f"Step {traj_idx[i]}", fontsize = 20)

    plt.tight_layout()

    return fig

def get_traj_fig(trajectory : torch.Tensor, num_points : int):
    """
    Plot the trajectory of 2d points.
    Makes 5 panes with the trajectory at the start, 1/4, 1/2, 3/4, and the end.
    """
    random_points = random.sample(range(num_points), 5)
    traj_len = trajectory.shape[0]
    traj_idx = [0, traj_len//4, traj_len//2, 3*traj_len//4, traj_len-1]
    trajectory_to_plot = trajectory[traj_idx, :, :]
    delta = 0.1
    min_x, max_x = trajectory[:, :, 0].min(), trajectory[:, :, 0].max()
    min_y, max_y = trajectory[:, :, 1].min(), trajectory[:, :, 1].max()
    min_x, max_x = min_x - delta, max_x + delta
    min_y, max_y = min_y - delta, max_y + delta

    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    colors = torch.sqrt(trajectory[0, :, 0] ** 2 + trajectory[0, :, 1] ** 2).tolist()
    for i in range(5):
        x, y = trajectory_to_plot[i, :, 0].cpu(), trajectory_to_plot[i, :, 1].cpu()
        ax[i].scatter(x, y, s = 1, c = colors, cmap = "viridis")
        
        x, y = trajectory_to_plot[i, random_points, 0].cpu(), trajectory_to_plot[i, random_points, 1].cpu()
        for j in range(len(random_points)):
            ax[i].text(x[j] + 0.1, y[j] + 0.1, j + 1, fontsize=10, color = "black", bbox=dict(facecolor='white', alpha=0.9))
        ax[i].scatter(x, y, s = 20, c = "black")
        
        ax[i].set_aspect('equal')
        ax[i].set_title(f"Step {traj_idx[i]}")
        ax[i].set_xlim(min_x, max_x)
        ax[i].set_ylim(min_y, max_y)
    
    plt.tight_layout()
    
    return fig

def get_grid_fig(left_images : torch.Tensor, right_images : torch.Tensor, num_rows : int) -> plt.Figure:
    """
    Makes two grids, one with the left images and one with the right images.
    """
    fig, axs = plt.subplots(num_rows, 2 * num_rows + 1, figsize=(20, 10))
    cmap = "gray" if left_images[0].shape[2] == 1 else None

    for i in range(num_rows ** 2):
        ax = axs[i // num_rows, i % num_rows]
        ax.imshow(left_images[i], cmap = cmap)

        ax = axs[i // num_rows, (i % num_rows) + num_rows + 1]
        ax.imshow(right_images[i], cmap = cmap)

    for ax in axs.flatten():
        ax.axis("off")

    axs[0, 0].set_title("Original", fontsize=16)
    axs[0, num_rows + 1].set_title("Generated", fontsize=16)

    plt.tight_layout()
    return fig