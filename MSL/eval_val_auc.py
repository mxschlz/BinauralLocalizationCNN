import numpy as np
import glob
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("TkAgg")

# get all auc files --> 10 nets, 10 files
files = sorted(glob.glob("*val_auc.npy"))
# get all netnames
netnames = [f.split("_val")[0] for f in files]

d = dict()

for file, net in zip(files, netnames):  # iterate through files
    data = np.load(file, allow_pickle=True)  # load data
    d[net] = []
    mv_num = list(data.item().keys())  # get all model versions from one net
    for i, mv in enumerate(mv_num):  # iterate through model versions
        mean = np.mean(data.item()[mv])
        d[net].append(mean)  # put average
df = pd.DataFrame(d)


df.plot()
plt.scatter(x=df.index, y=df.mean(axis=1))
plt.title("AUC")
plt.show()
