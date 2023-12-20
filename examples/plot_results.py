from CNN_result_analysis import *
import seaborn as sns
import matplotlib.pyplot as plt

res_path = 'Result'
res_patt = os.path.join(res_path, 'broadband_noise_elevation_net1_*csv')

header, data = read_resfiles(res_patt)
data = np.concatenate(data, axis=0)

col_act = header.index('train/cnn_idx')
pos_act = CNNpos_to_loc(data[:, col_act])[1]
pos_act[pos_act > 180] = pos_act[pos_act > 180] - 360

col_pred = header.index('model_pred')
pos_pred = CNNpos_to_loc(data[:, col_pred])[1]
pos_pred[pos_pred > 180] = pos_pred[pos_pred > 180] - 360

# collapse front and back
pos_pred[pos_pred > 90] = 180 - pos_pred[pos_pred > 90]
pos_pred[pos_pred < -90] = -180 - pos_pred[pos_pred < -90]

sns.regplot(pos_act, pos_pred)