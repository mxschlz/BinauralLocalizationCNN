from CNN_result_analysis import *
from MSL.config_MSL import CONFIG_TEST as cfg
import seaborn as sns
import scienceplots
plt.style.use("science")
import os
from analysis_and_plotting.decision_rule import decide_sound_presence
plt.ion()


# get model versions
models = cfg["DEFAULT_RUN_PARAM"]["model_version"]


# results directory
results_root = "Results_no_augment_forward"

# read results files from csv
model = models[7]
#for model in models:
csv_patt = os.path.join(results_root, f"*_ele*model_{model}.csv")
header, csv = read_resfiles(csv_patt, filetype="csv")
csv = np.concatenate(csv, axis=0)
col_act = header.index('train/n_sounds')  # get actual position column
n_act = csv[:, col_act]

# get cond_dist from all nets
npy_patt = os.path.join(results_root, f"*_ele*model_{model}_cd*.npy")
npy = read_resfiles(npy_patt, filetype="npy")
npy = np.concatenate(npy, axis=1)
# crit_range = np.linspace(0.08, 0.09, 11)
# for crit_val in crit_range:
to_plot = list()
idx_start = 0
for i, d in enumerate(npy):  # TODO: original cd shape of one run is (16, 504)
    n_sounds_perceived = decide_sound_presence(d, criterion=0.085)
    act = n_act[idx_start:idx_start+len(n_sounds_perceived)].tolist()
    idx_start += len(n_sounds_perceived)
    to_plot.append([act, n_sounds_perceived])
to_plot = np.array(to_plot)
to_plot = to_plot.transpose(0, 2, 1).reshape((4960, 2))

new_x = list()
new_y = list()
for x, y in zip(to_plot[:, 0], to_plot[:, 1]):
    if x != 1:
        new_x.append(x)
        new_y.append(y)
    else:
        continue


sns.lineplot(x=new_x, y=new_y, err_style="bars", label="Forward Speech", errorbar=("se", 2))

# results directory
results_root = "Results_no_augment_reversed"

# read results files from csv
model = models[10]
#for model in models:
csv_patt = os.path.join(results_root, f"*_ele*model_{model}.csv")
header, csv = read_resfiles(csv_patt, filetype="csv")
csv = np.concatenate(csv, axis=0)
col_act = header.index('train/n_sounds')  # get actual position column
n_act = csv[:, col_act]

# get cond_dist from all nets
npy_patt = os.path.join(results_root, f"*_ele*model_{model}_cd*.npy")
npy = read_resfiles(npy_patt, filetype="npy")
npy = np.concatenate(npy, axis=1)
# crit_range = np.linspace(0.08, 0.09, 11)
# for crit_val in crit_range:
to_plot = list()
idx_start = 0
for i, d in enumerate(npy):  # TODO: original cd shape of one run is (16, 504)
    n_sounds_perceived = decide_sound_presence(d, criterion=0.08)
    act = n_act[idx_start:idx_start+len(n_sounds_perceived)].tolist()
    idx_start += len(n_sounds_perceived)
    to_plot.append([act, n_sounds_perceived])
to_plot = np.array(to_plot)
to_plot = to_plot.transpose(0, 2, 1).reshape((4960, 2))

new_x = list()
new_y = list()
for x, y in zip(to_plot[:, 0], to_plot[:, 1]):
    if x != 1:
        new_x.append(x)
        new_y.append(y)
    else:
        continue

sns.lineplot(x=new_x, y=new_y, err_style="bars", label="Reversed Speech", errorbar=("se", 2))
# sns.lineplot(x=to_plot[:, 0], y=to_plot[:, 1], err_style="bars", ax=ax[1])

plt.xlim([1.5, 6.5])
plt.ylim([1.5, 6.5])
#plt.xticks([2,3,4,5,6], [6,5,4,3,2,])
plt.xlabel("Actual Number Of Sounds")
plt.ylabel("Reported Number Of Sounds")
#plt.gca().invert_xaxis()
plt.plot(plt.xlim(), plt.ylim(), ls="--", c=".3")
plt.show()