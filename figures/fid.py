import pandas as pd
import matplotlib.pyplot as plt

# load a csv file
df = pd.read_csv('figures/fid.csv')
fid = df["good results - benchmarks/FID"]
# plot the data on a log scale
plt.plot(fid, linestyle='-.')
plt.xticks(range(0, 20, 2))
plt.xlabel('DSB iteration')
plt.ylabel('FID')
plt.savefig('figures/figures/fid.png')