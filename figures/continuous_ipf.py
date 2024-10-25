import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define grid for plotting
x, y = np.linspace(-5, 5, 100), np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Define two Gaussian distributions
mean1 = np.array([-2, -2])
cov1 = np.array([[1, 0], [0, 1]])  # Standard normal distribution
rv1 = multivariate_normal(mean1, cov1)

mean2 = np.array([2, 2])
cov2 = np.array([[1, 0.5], [0.5, 1]])  # Slightly rotated distribution
rv2 = multivariate_normal(mean2, cov2)

# Calculate the density
Z1 = rv1.pdf(pos)
Z2 = rv2.pdf(pos)

# Plot the contour plots
plt.figure(figsize=(8, 6))
contour1 = plt.contour(X, Y, Z1, colors='blue')
contour2 = plt.contour(X, Y, Z2, colors='red')
# write the text p_prior and p_data beside the contour
plt.text(-4, 0, r'$p_{\text{data}}$', color='blue')
plt.text(3, 0, r'$p_{\text{prior}}$', color='red')

# draw 5 random arrows that go from pdata to somewhere random
center_start = mean1
center_end = np.array([-2, 4])
direction = center_end - center_start
for _ in range(10):
    # make the start around the center
    start = center_start + np.random.randn(2) * 0.5
    plt.arrow(*start, *direction, head_width=0.2, head_length=0.5, fc='black')

plt.text(-4, 3, r'$\pi^0=p^0$', color='black')
    
# Add labels and legend
plt.xlabel('$x_0$')
plt.ylabel('$x_1$')
plt.xticks([])
plt.yticks([])
plt.savefig("figures/figures/ipf_p0.png")