import matplotlib
matplotlib.use("TkAgg")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="white", palette="viridis")
from MSL.CNN_results_plot import read_resfiles
from stim_util import CNNpos_to_loc
import os
import numpy as np

res_path = 'Result'
model_data_pattern = os.path.join(res_path, "LocaAccu_babble_ele_result_net1*")


header, data = read_resfiles(model_data_pattern)  # read results file
data = np.concatenate(data, axis=0)  # concatenate data
col_act = header.index('train/cnn_idx')  # get actual position column
col_pred = header.index('model_pred')  # get predicted position
loc_act = CNNpos_to_loc(data[:, col_act])[1]  # convert bin to azi, ele positions
loc_pred = CNNpos_to_loc(data[:, col_pred])[1]
sns.regplot(loc_act, loc_pred)