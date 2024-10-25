import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 1, 100)

def make_random_path(t):
    path = np.ones_like(t)
    for i in range(1, len(t)):
        path[i] = path[i-1] + np.random.randn() * 0.1
    return path

for _ in range(100):
    path = make_random_path(t)
    lw = min(1 / path[-1], 2)
    plt.plot(t, path, lw=lw)

plt.xlabel('$t$')
plt.ylabel('$x(t)$')
plt.savefig('figures/figures/paths.png', dpi=300, bbox_inches='tight')