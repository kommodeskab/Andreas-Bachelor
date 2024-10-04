import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

t_points = 100
x_points = 100
t = np.linspace(0, 1, t_points)
x = np.linspace(-5, 5, x_points)
T, X = np.meshgrid(t, x)
sigma = 0.5 + T  
mu = np.sin(T * 10) * 2 
Z = np.exp(-(X - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    
plt.figure(figsize=(12, 8))
plt.pcolormesh(t, x, Z, shading='auto')
plt.colorbar(label='Probability Density')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Time-Varying Probability Distribution')
plt.savefig("figures/figures/continousDSB.png")