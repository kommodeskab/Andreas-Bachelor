import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate some data points
data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=1000)
data_scewed = data.copy()
# for positive y-values, move them to the right. The higher the y-value, the further to the right
mask = data_scewed[:,1] > 0
data_scewed[mask,0] += data_scewed[mask,1] ** 2

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Create a grid of points
x = np.linspace(-3, 6, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
positions = np.vstack([X.ravel(), Y.ravel()])

# Perform kernel density estimation
kde_1 = gaussian_kde(data.T)
Z_1 = np.reshape(kde_1(positions).T, X.shape)

kde_2 = gaussian_kde(data_scewed.T)
Z_2 = np.reshape(kde_2(positions).T, X.shape)

# Create the contour plot
axs[0].contour(X, Y, Z_1, levels=10)
axs[1].contour(X, Y, Z_2, levels=10)
axs[0].set_title('No drug administered')
axs[1].set_title('Drug administered')
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Blood pressure')
    ax.set_ylabel('BMI')
plt.tight_layout()

plt.savefig("figures/figures/example_case.png", dpi=300, bbox_inches='tight')