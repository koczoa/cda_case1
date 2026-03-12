import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

colorscheme = "viridis"
df = pd.read_csv("case1Data.csv")

prod_cols = df.columns

num_nodes = len(prod_cols)
matrix = df.loc[:, prod_cols].corr().to_numpy()

upper_mask = np.tri(matrix.shape[0], matrix.shape[1], k=-1).T
masked = np.ma.array(matrix, mask=upper_mask)

# fig, ax = plt.subplots(figsize=(10, 10))
#
# im = ax.imshow(masked, cmap=colorscheme, interpolation='nearest')
#
# ax.set_xticks(np.arange(num_nodes))
# ax.set_xticklabels(prod_cols, rotation=45, ha='right', wrap=True)
#
# ax.set_yticks(np.arange(num_nodes))
# ax.set_yticklabels(prod_cols)
#
# fig.colorbar(im, ax=ax, shrink=0.6)
# plt.title("Correlation Matrix")
#
# fig.tight_layout()
# plt.show()


print("Top Absolute Correlations for y compared to others")
c = df.corr()['y'].abs().sort_values(ascending=False)
print(c[:20])
# so = c.sort_values(kind="quicksort", ascending=True)
# print("Highest absolute correlation")
# print(so[100:120])
#
# print(f"number of NaN's: {so.isna().sum()}")

