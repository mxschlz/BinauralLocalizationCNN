import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
# plt.style.use("science")
plt.ion()
from CNN_result_analysis import read_resfiles
from stim_util import CNNpos_to_loc
import os
import numpy as np
import matplotlib

x = [1,2,3,4,5,6,7]
y = [1,2,3,4,5,6,7]
plt.plot(x, y)
# plt.savefig("/home/max/PycharmProjects/BinauralLocalizationCNN/plots/fig.png", ppi=400)
