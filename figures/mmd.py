import pandas as pd
import matplotlib.pyplot as plt

# load a csv file
df = pd.read_csv('figures/mmd_values.csv')
tr = df["TR - benchmarks/MMD"]
fr = df["FR - benchmarks/MMD"]
baseline = df["TR - benchmarks/MMD_baseline"] # the baseline is the same for both TR and FR
# plot the data on a log scale
plt.plot(tr, label="TR")
plt.plot(fr, label="FR")
plt.plot(baseline, label="baseline", linestyle='--')
plt.legend()
plt.xlabel('DSB iteration')
plt.ylabel('MMD')
plt.savefig('figures/figures/mmd.png')