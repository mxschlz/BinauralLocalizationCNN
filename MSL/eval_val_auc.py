import numpy as np
import glob
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
plt.style.use("science")
plt.ion()
import seaborn as sns

matplotlib.use("TkAgg")

# get all auc files --> 10 nets, 10 files
files = sorted(glob.glob("val_auc/*val_auc.npy"))
# get all netnames
netnames = [f.split("/")[1].split("_val")[0] for f in files]

d = dict()

for file, net in zip(files, netnames):  # iterate through files
    data = np.load(file, allow_pickle=True)  # load data
    d[net] = []
    mv_num = list(data.item().keys())  # get all model versions from one net
    for i, mv in enumerate(mv_num):  # iterate through model versions
        mean = np.mean(data.item()[mv])
        d[net].append(mean)  # put average
df = pd.DataFrame(d)


x = df.index
y = df.values
df.plot(alpha=0.3, legend="")
plt.plot(df.index, df.mean(axis=1), color="black")
sns.scatterplot(x=df.index, y=df.mean(axis=1), color="black", marker="X", legend="")
plt.vlines(x=14, ymin=plt.ylim()[0], ymax=df.mean(axis=1)[14], color="grey", linestyles="dashed")
plt.hlines(y=df.mean(axis=1)[14], xmin=plt.xlim()[0], xmax=14, color="grey", linestyles="dashed")
plt.xlabel("Training Level Per Hundred Steps")
plt.ylabel("Area Under The Curve [Percent]")
