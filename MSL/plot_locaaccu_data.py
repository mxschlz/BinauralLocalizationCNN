import matplotlib
# .use("Qt5Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
plt.style.use("science")
plt.ion()
from CNN_result_analysis import read_resfiles
from stim_util import CNNpos_to_loc
import os
import numpy as np
import string


res_path = 'Results_locaaccu'
model_data_pattern_noise = os.path.join(res_path, "*noise_azi_*")
model_data_pattern_babble = os.path.join(res_path, "*babble_azi_*")

layout = """
ab
"""
ax = plt.figure().subplot_mosaic(layout, sharex=True, sharey=True)

header, data = read_resfiles(model_data_pattern_noise)  # read results file
data = np.concatenate(data, axis=0)  # concatenate data
col_act = header.index('train/cnn_idx')  # get actual position column
col_pred = header.index('model_pred')  # get predicted position
loc_act = CNNpos_to_loc(data[:, col_act])[0]  # convert bin to azi, ele positions
loc_pred = CNNpos_to_loc(data[:, col_pred])[0]
loc_act[loc_act > 180] = loc_act[loc_act > 180] - 360
loc_pred[loc_pred > 180] = loc_pred[loc_pred > 180] - 360
# collapse front and back
loc_pred[loc_pred > 90] = 180 - loc_pred[loc_pred > 90]
loc_pred[loc_pred < -90] = -180 - loc_pred[loc_pred < -90]
sns.lineplot(x=loc_act, y=loc_pred, ax=ax["a"], label="Pink Noise")

header, data = read_resfiles(model_data_pattern_babble)  # read results file
data = np.concatenate(data, axis=0)  # concatenate data
col_act = header.index('train/cnn_idx')  # get actual position column
col_pred = header.index('model_pred')  # get predicted position
loc_act = CNNpos_to_loc(data[:, col_act])[0]  # convert bin to azi, ele positions
loc_pred = CNNpos_to_loc(data[:, col_pred])[0]
loc_act[loc_act > 180] = loc_act[loc_act > 180] - 360
loc_pred[loc_pred > 180] = loc_pred[loc_pred > 180] - 360
# collapse front and back
loc_pred[loc_pred > 90] = 180 - loc_pred[loc_pred > 90]
loc_pred[loc_pred < -90] = -180 - loc_pred[loc_pred < -90]
sns.lineplot(x=loc_act, y=loc_pred, ax=ax["a"], label="Babble Noise")
#ax["a"].legend(["Pink Noise", "Babble Noise"])

model_data_pattern_noise = os.path.join(res_path, "*noise_ele_*")
model_data_pattern_babble = os.path.join(res_path, "*babble_ele_*")
header, data = read_resfiles(model_data_pattern_noise)  # read results file
data = np.concatenate(data, axis=0)  # concatenate data
col_act = header.index('train/cnn_idx')  # get actual position column
col_pred = header.index('model_pred')  # get predicted position
loc_act = CNNpos_to_loc(data[:, col_act])[1]  # convert bin to azi, ele positions
loc_pred = CNNpos_to_loc(data[:, col_pred])[1]
loc_act[loc_act > 180] = loc_act[loc_act > 180] - 360
loc_pred[loc_pred > 180] = loc_pred[loc_pred > 180] - 360
# collapse front and back
loc_pred[loc_pred > 90] = 180 - loc_pred[loc_pred > 90]
loc_pred[loc_pred < -90] = -180 - loc_pred[loc_pred < -90]
sns.lineplot(x=loc_act-20, y=loc_pred-20, ax=ax["b"], label="Pink Noise")

header, data = read_resfiles(model_data_pattern_babble)  # read results file
data = np.concatenate(data, axis=0)  # concatenate data
col_act = header.index('train/cnn_idx')  # get actual position column
col_pred = header.index('model_pred')  # get predicted position
loc_act = CNNpos_to_loc(data[:, col_act])[1]  # convert bin to azi, ele positions
loc_pred = CNNpos_to_loc(data[:, col_pred])[1]
loc_act[loc_act > 180] = loc_act[loc_act > 180] - 360
loc_pred[loc_pred > 180] = loc_pred[loc_pred > 180] - 360
# collapse front and back
loc_pred[loc_pred > 90] = 180 - loc_pred[loc_pred > 90]
loc_pred[loc_pred < -90] = -180 - loc_pred[loc_pred < -90]
sns.lineplot(x=loc_act-20, y=loc_pred-20, ax=ax["b"], label="Babble Noise")
#ax["b"].legend(["Pink Noise", "Babble Noise"])

ax["a"].plot(ax["a"].get_xlim(), ax["a"].get_ylim(), ls="--", c=".3")
ax["b"].plot(ax["b"].get_xlim(), ax["b"].get_ylim(), ls="--", c=".3")
ax["a"].text(-0.1, 1.0, string.ascii_uppercase[0], transform=ax["a"].transAxes,
               size=20, weight='bold')
ax["b"].text(-0.1, 1.0, string.ascii_uppercase[1], transform=ax["b"].transAxes,
               size=20, weight='bold')
plt.savefig("/home/max/PycharmProjects/BinauralLocalizationCNN/plots/locaaccu_model_performance.png",
            dpi=400,
            bbox_inches="tight")

