import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from numpy.random import normal

def normal_distribution(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

num_slices = 20
plt.figure(figsize=(12, 8))

# Space points
x = np.linspace(-5, 5, 1000)

# Parameters that vary with time
max_height = 0.5  # Maximum height for scaling
xn = 1.5 # start value
max_xn, min_xn = 1.5, 1.5
std = 0.2

for i in range(num_slices):
    xn = xn - 1 / num_slices * xn
    
    # sample new xn from normal distribution
    xn = normal(xn, std, 1)
    # make a horizontal line
    if i > 0:
        plt.plot([i-1, i], [xn, xn], 'r-', alpha=0.5)
    
    if xn > max_xn:
        max_xn = xn
    if xn < min_xn:
        min_xn = xn
    
    # Calculate the distribution
    dist = normal_distribution(x, xn, std)
    
    # Scale the distribution for better visualization
    scaled_dist = dist * max_height / dist.max()
    
    # Create the polygon for filled distribution
    xy = np.column_stack([scaled_dist + i, x])
    polygon = Polygon(xy, facecolor='blue', alpha=0.3)
    plt.gca().add_patch(polygon)
    
    # Plot the distribution outline
    plt.plot(scaled_dist + i, x, 'b-', alpha=0.5)

# Set plot limits and labels
plt.xlim(-0.5, num_slices - 0.5)
plt.ylim(min_xn - 1, max_xn + 1)

# Set custom x-ticks to show time values
plt.xticks(range(num_slices), ["$x_{%i}$" % i for i in range(num_slices)])
plt.xlabel('Time')
plt.ylabel('Position')
plt.savefig("figures/figures/markov_chain.png", dpi=300, bbox_inches="tight")