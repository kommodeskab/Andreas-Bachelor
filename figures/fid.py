import pandas as pd
import matplotlib.pyplot as plt

# load a csv file
df = pd.read_csv('figures/fid.csv')
fid_tr = df["TR - benchmarks/FID"]
fid_fr = df["FR - benchmarks/FID"]
# plot the data on a log scale
plt.plot(fid_tr, label='TR')
plt.plot(fid_fr, label='FR')
plt.xticks(range(0, 20, 2))
plt.yticks(range(6, 30, 2))
plt.xlabel('DSB iteration')
plt.ylabel('FID')
plt.legend()
plt.savefig('figures/figures/fid.png', dpi=300, bbox_inches='tight')