from CNN_result_analysis import *
from MSL.config_MSL import CONFIG_TEST as cfg
import seaborn as sns
import matplotlib
import os
matplotlib.use("TkAgg")


# get model versions
models = cfg["DEFAULT_RUN_PARAM"]["model_version"]

# results directory
results_root = "Result"

# read results files from csv
res_patt = os.path.join(results_root, "test_model_net1_50*")
header, data = read_resfiles(res_patt)
col_act = header.index('train/n_sounds')  # get actual position column
col_pred = header.index('model_pred')  # get predicted position
n_act = data[0][:, col_act]
n_pred = CNNpos_to_n_sounds(data[0][:, col_pred])
sns.lineplot(x=n_act, y=n_pred)
plt.show()