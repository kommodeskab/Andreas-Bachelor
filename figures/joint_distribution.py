from seaborn import jointplot
import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(0)
x = np.random.normal(size=1000)
y = 2 * x + np.random.normal(size=1000)

jointplot(x=x, y=y, kind='kde')
plt.xlabel("$x_0$")
plt.ylabel("$x_1$")
plt.text(-0.5, 8, r"$p_{data}$", fontsize=10)
# rotate the text
plt.text(3.5, -0.5, r"$p_{prior}$", fontsize=10, rotation=90)
plt.savefig("figures/figures/joint_distribution.png", dpi=300, bbox_inches="tight")