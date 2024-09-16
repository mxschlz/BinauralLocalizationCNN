import matplotlib

matplotlib.use("TkAgg")
from CNN_result_analysis import result_figure1, result_figure2, result_figure3
import os

res_path = 'Result'
# figure 1
res_patt = os.path.join(res_path, 'broadband_noise_azimuth_net1*csv')
hdata_path = os.path.join(res_path, 'human_f1.xlsx')
result_figure1(res_patt, hdata_path)

# figure 2
# no human data available
res_patt_2 = os.path.join(res_path, 'ITDILD_*_100000.csv')
result_figure2(res_patt_2)

# figure 3
res_patt_3 = os.path.join(res_path, 'noise_bw_*.csv')
hdata_path_3 = os.path.join(res_path, 'human_f3.xlsx')
result_figure3(res_patt_3, hdata_path_3)
