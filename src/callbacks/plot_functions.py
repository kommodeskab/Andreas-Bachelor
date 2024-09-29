import matplotlib.pyplot as plt
import random
import torch
import matplotlib

matplotlib.use('Agg')

def get_gamma_fig(gammas, ylabel, title):
    ts = range(1, len(gammas)+1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts, gammas)
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig

def get_image_fig(trajectory : torch.Tensor, title : str):
    traj_len = trajectory.shape[0]
    traj_idx = [0, traj_len//4, traj_len//2, 3*traj_len//4, traj_len-1]
    cmap = "gray" if trajectory.size(2) == 1 else None

    fig, ax = plt.subplots(5, 5, figsize=(20, 20))
    for i in range(5): 
        for j in range(5): 
            img = trajectory[traj_idx[i], j, :, :, :].permute(1, 2, 0).cpu()
            ax[i, j].imshow(img, cmap = cmap)
            ax[i, j].axis("off")
            if j == 2:
                ax[i, j].set_title(f"Step {traj_idx[i]}", fontsize = 20)

    fig.suptitle(title, fontsize = 40)
    return fig

def get_traj_fig(trajectory : torch.Tensor, title : str, num_points : int):
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
    
    fig.suptitle(title, fontsize = 20)
    
    return fig