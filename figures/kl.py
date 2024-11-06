import pandas as pd
import matplotlib.pyplot as plt

kl_data = pd.read_csv("figures/kl.csv")
dims = [16,32,64,256]
colors = ['r', 'g', 'b', 'y']
for color, dim in zip(colors, dims):
    kl = kl_data[f"dim: {dim} - benchmarks/kl__MIN"]
    baseline = kl_data[f"dim: {dim} - benchmarks/baseline_kl"]
    plt.plot(kl, label=f"dim: {dim}", color=color)
    plt.plot(baseline, label=f"baseline dim: {dim}", linestyle='--', color=color)
plt.legend()
plt.xlim(0, 20)
#only show every 5th tick
plt.xticks(range(0, 21, 5))
plt.yscale('log')
plt.xlabel('DSB iteration')
plt.ylabel('KL')
plt.savefig('figures/figures/kl.png')