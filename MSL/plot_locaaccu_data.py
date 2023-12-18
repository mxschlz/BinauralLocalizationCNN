import matplotlib
matplotlib.use("TkAgg")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="white", palette="viridis")
from MSL.CNN_results_plot import results_locaaccu
import os

res_path = 'Result'

model_data_pattern = os.path.join(res_path, "LocaAccu_babble_azi_result*")
human_data_path = "human_bevahiour/locaaccu_h.hdf"
results_locaaccu(model_data_pattern, human_data_path, plane="h", stimtype="babble", plot_type="reg")

model_data_pattern = os.path.join(res_path, "LocaAccu_noise_azi_result*")
human_data_path = "human_bevahiour/locaaccu_h.hdf"
results_locaaccu(model_data_pattern, human_data_path, plane="h", stimtype="noise", plot_type="reg")

model_data_pattern = os.path.join(res_path, "LocaAccu_babble_ele_result*")
human_data_path = "human_bevahiour/locaaccu_h.hdf"
results_locaaccu(model_data_pattern, human_data_path, plane="v", stimtype="babble", plot_type="reg")

model_data_pattern = os.path.join(res_path, "LocaAccu_noise_ele_result*")
human_data_path = "human_bevahiour/locaaccu_v.hdf"
results_locaaccu(model_data_pattern, human_data_path, plane="v", stimtype="babble", plot_type="reg")
