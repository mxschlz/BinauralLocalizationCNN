from CNN_result_analysis import *
import seaborn as sns
import matplotlib.pyplot as plt

res_path = 'Result'
res_patt_azi = os.path.join(res_path, 'broadband_noise_azimuth_net5_*csv')
res_patt_ele = os.path.join(res_path, 'broadband_noise_elevation_net5_*csv')
fig, ax = plt.subplots(1, 2)

header, data = read_resfiles(res_patt_azi)
data = np.concatenate(data, axis=0)

col_act = header.index('train/cnn_idx')
pos_act = CNNpos_to_loc(data[:, col_act])[0]
pos_act[pos_act > 180] = pos_act[pos_act > 180] - 360

col_pred = header.index('model_pred')
pos_pred = CNNpos_to_loc(data[:, col_pred])[0]
pos_pred[pos_pred > 180] = pos_pred[pos_pred > 180] - 360

# collapse front and back
pos_pred[pos_pred > 90] = 180 - pos_pred[pos_pred > 90]
pos_pred[pos_pred < -90] = -180 - pos_pred[pos_pred < -90]

sns.lineplot(pos_act, pos_pred, ax=ax[0])

# elevation
header, data = read_resfiles(res_patt_ele)
data = np.concatenate(data, axis=0)
col_act = header.index('train/cnn_idx')
pos_act = CNNpos_to_loc(data[:, col_act])[1]
col_pred = header.index('model_pred')
pos_pred = CNNpos_to_loc(data[:, col_pred])[1]
sns.lineplot(pos_act, pos_pred, ax=ax[1])

ax[0].plot(ax[0].get_xlim(), ax[0].get_ylim())
ax[1].plot(ax[1].get_xlim(), ax[1].get_ylim())
plt.show()
