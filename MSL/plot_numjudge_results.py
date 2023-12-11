from CNN_result_analysis import *
from MSL.config_MSL import CONFIG_TEST as cfg
import seaborn as sns
import matplotlib
import os
from analysis_and_plotting.decision_rule import decide_sound_presence
matplotlib.use("TkAgg")


# get model versions
models = cfg["DEFAULT_RUN_PARAM"]["model_version"]

# results directory
results_root = "Result"

# read results files from csv
model = models[0]
for model in models:
    print(f"Model {model} output: ")
    csv_patt = os.path.join(results_root, f"*_azi*model_{model}.csv")
    header, csv = read_resfiles(csv_patt, filetype="csv")
    csv = np.concatenate(csv, axis=0)
    col_act = header.index('train/n_sounds')  # get actual position column
    col_pred = header.index('model_pred')  # get predicted position
    n_act = csv[:, col_act]
    n_pred = csv[:, col_pred]

    # get cond_dist from all nets
    npy_patt = os.path.join(results_root, f"*_azi*model_{model}_cd*.npy")
    npy = read_resfiles(npy_patt, filetype="npy")
    npy = np.concatenate(npy, axis=1)
    crit_range = np.linspace(0, 1, 11)
    # for crit_val in crit_range:
    to_plot = list()
    idx_start = 0
    for i, d in enumerate(npy):  # TODO: original cd shape of one run is (16. 504)
        n_sounds_perceived = decide_sound_presence(d, criterion=0.09)
        act = n_act[idx_start:idx_start+len(n_sounds_perceived)].tolist()
        idx_start += len(n_sounds_perceived)
        to_plot.append([act, n_sounds_perceived])
    to_plot = np.array(to_plot)
    to_plot = to_plot.transpose(0, 2, 1).reshape((4960, 2))

    # sns.lineplot(x=n_act, y=n_pred, err_style="bars")
    sns.lineplot(x=to_plot[:, 0], y=to_plot[:, 1], err_style="bars")
    # plt.xlim([1, 6])
    # plt.ylim([1, 6])
    plt.plot(plt.xlim(), plt.ylim(), ls="--", c=".3")
    plt.show(block=True)
