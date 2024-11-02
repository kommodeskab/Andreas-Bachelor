import numpy as np
import matplotlib.pyplot as plt

# Parameters for the vector field
grid_size = 20  # Number of points in each dimension
noise_scale = 0.05  # Scale of the noise to simulate Wiener process randomness
x_range = y_range = np.linspace(-2, 2, grid_size)  # Range for x and y axes
alpha = 0.2  # Coefficient for the deterministic part of the vector field

# Initialize grid for the vector field
X, Y = np.meshgrid(x_range, y_range)
U = - alpha * X  # Deterministic part of f(x, y) = - x (points towards the origin)
V = - alpha * Y  # Deterministic part of f(x, y) = - y (points towards the origin)

# Add random noise to simulate the stochastic component
U_noisy = U + noise_scale * np.random.normal(size=U.shape)
V_noisy = V + noise_scale * np.random.normal(size=V.shape)

# Plot the vector field
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U_noisy, V_noisy, color='b', angles='xy', scale_units='xy', scale=1)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.savefig("figures/figures/SDE.png")