import numpy as np
import glob
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("TkAgg")


files = sorted(glob.glob("*val_auc.npy"))


df = pd.DataFrame()

for file in files:
    data = np.load(file, allow_pickle=True)
    keys = list(data.item().keys())
    for key in keys:
        df[key] = pd.Series(np.mean(data.item()[key]))


df.boxplot()
plt.show()
