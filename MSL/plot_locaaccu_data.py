import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from MSL.CNN_results_plot import results_locaaccu
import os

res_path = 'Result'

# plot LocaAccu noise h
model_data_pattern = os.path.join(res_path, "locaaccu_noise_h*")
human_data_path = "tfrecords/msl/human_bevahiour/locaaccu_h.hdf"
results_locaaccu(model_data_pattern, human_data_path, plane="h", stimtype="noise", plot_type="raincloud")

# plot LocaAccu noise v
model_data_pattern = os.path.join(res_path, "locaaccu_noise_v*")
human_data_path = "tfrecords/msl/human_bevahiour/locaaccu_v.hdf"
results_locaaccu(model_data_pattern, human_data_path, plane="v", stimtype="noise", plot_type="raincloud")

# plot LocaAccu babble h
model_data_pattern = os.path.join(res_path, "locaaccu_babble_h*")
human_data_path = "tfrecords/msl/human_bevahiour/locaaccu_h.hdf"
results_locaaccu(model_data_pattern, human_data_path, plane="h", stimtype="babble", plot_type="raincloud")

# plot LocaAccu babble v
model_data_pattern = os.path.join(res_path, "locaaccu_babble_v*")
human_data_path = "tfrecords/msl/human_bevahiour/locaaccu_v.hdf"
results_locaaccu(model_data_pattern, human_data_path, plane="v", stimtype="babble", plot_type="raincloud")
