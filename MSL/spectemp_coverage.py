import pickle

import matplotlib.pyplot as plt
import numpy as np
import slab

from analysis_and_plotting.misc import spectemp_coverage

plt.style.use("science")
import seaborn as sns
import matplotlib.pyplot as plt

plt.ion()
import string

talkers_clear = pickle.load(open("/home/max/labplatform/sound_files/numjudge_talker_files_clear.pkl", "rb"))
talkers_reversed = pickle.load(open("/home/max/labplatform/sound_files/numjudge_talker_files_reversed.pkl", "rb"))

# some params important for spectemp coverage function
upper_freq = 11000
dyn_range = 65
resize = 0.6  # resize duration

signals_sample = slab.ResultsFile.read_file(filename="logfiles/log_train_test/log_train_test_2023-12-07-16-50-58.txt",
                                            tag="signals_sample_azi")
speakers_sample = slab.ResultsFile.read_file(filename="logfiles/log_train_test/log_train_test_2023-12-07-16-50-58.txt",
                                             tag="speakers_sample_azi")
country_idxs = slab.ResultsFile.read_file(filename="logfiles/log_train_test/log_train_test_2023-12-07-16-50-58.txt",
                                          tag="country_idxs_azi")

# gather all available stimuli
stimlist_clear = dict()
for talker in list(talkers_clear.keys()):
    stimlist_clear[talker] = list()

    for stim in talkers_clear[talker]:
        # stim = stim.resample(samplerate)
        stimlist_clear[talker].append(stim)

stimlist_reversed = dict()
for talker in list(talkers_reversed.keys()):
    stimlist_reversed[talker] = list()

    for stim in talkers_reversed[talker]:
        # stim = stim.resample(samplerate)
        stimlist_reversed[talker].append(stim)

data_clear = dict(n_sounds=[], coverage=[])
data_reversed = dict(n_sounds=[], coverage=[])

for signals, speakers, countries in zip(signals_sample, speakers_sample, country_idxs):
    n_sounds = len(speakers)
    sound = slab.Sound(data=np.zeros(48828), samplerate=48828)
    trial_composition = [stimlist_clear[x][y].resize(resize) for x, y in zip(signals, countries)]
    percentage_filled = spectemp_coverage(trial_composition, dyn_range=dyn_range, upper_freq=upper_freq)
    data_clear["coverage"].append(percentage_filled)
    data_clear["n_sounds"].append(n_sounds)

    sound = slab.Sound(data=np.zeros(48828), samplerate=48828)
    trial_composition = [stimlist_reversed[x][y].resize(resize) for x, y in zip(signals, countries)]
    percentage_filled = spectemp_coverage(trial_composition, dyn_range=dyn_range, upper_freq=upper_freq)
    data_reversed["coverage"].append(percentage_filled)
    data_reversed["n_sounds"].append(n_sounds)

new_x_clear_azi = list()
new_y_clear_azi = list()
for x, y in zip(data_clear["n_sounds"], data_clear["coverage"]):
    if x != 1:
        new_x_clear_azi.append(x)
        new_y_clear_azi.append(y)
    else:
        continue

new_x_rev_azi = list()
new_y_rev_azi = list()
for x, y in zip(data_reversed["n_sounds"], data_reversed["coverage"]):
    if x != 1:
        new_x_rev_azi.append(x)
        new_y_rev_azi.append(y)
    else:
        continue

signals_sample = slab.ResultsFile.read_file(filename="logfiles/log_train_test/log_train_test_2023-12-07-16-50-58.txt",
                                            tag="signals_sample_ele")
speakers_sample = slab.ResultsFile.read_file(filename="logfiles/log_train_test/log_train_test_2023-12-07-16-50-58.txt",
                                             tag="speakers_sample_ele")
country_idxs = slab.ResultsFile.read_file(filename="logfiles/log_train_test/log_train_test_2023-12-07-16-50-58.txt",
                                          tag="country_idxs_ele")

# gather all available stimuli
stimlist_clear = dict()
for talker in list(talkers_clear.keys()):
    stimlist_clear[talker] = list()

    for stim in talkers_clear[talker]:
        # stim = stim.resample(samplerate)
        stimlist_clear[talker].append(stim)

stimlist_reversed = dict()
for talker in list(talkers_reversed.keys()):
    stimlist_reversed[talker] = list()

    for stim in talkers_reversed[talker]:
        # stim = stim.resample(samplerate)
        stimlist_reversed[talker].append(stim)

data_clear = dict(n_sounds=[], coverage=[])
data_reversed = dict(n_sounds=[], coverage=[])

for signals, speakers, countries in zip(signals_sample, speakers_sample, country_idxs):
    n_sounds = len(speakers)
    sound = slab.Sound(data=np.zeros(48828), samplerate=48828)
    trial_composition = [stimlist_clear[x][y].resize(resize) for x, y in zip(signals, countries)]
    percentage_filled = spectemp_coverage(trial_composition, dyn_range=dyn_range, upper_freq=upper_freq)
    data_clear["coverage"].append(percentage_filled)
    data_clear["n_sounds"].append(n_sounds)

    sound = slab.Sound(data=np.zeros(48828), samplerate=48828)
    trial_composition = [stimlist_reversed[x][y].resize(resize) for x, y in zip(signals, countries)]
    percentage_filled = spectemp_coverage(trial_composition, dyn_range=dyn_range, upper_freq=upper_freq)
    data_reversed["coverage"].append(percentage_filled)
    data_reversed["n_sounds"].append(n_sounds)

new_x_clear_ele = list()
new_y_clear_ele = list()
for x, y in zip(data_clear["n_sounds"], data_clear["coverage"]):
    if x != 1:
        new_x_clear_ele.append(x)
        new_y_clear_ele.append(y)
    else:
        continue

new_x_rev_ele = list()
new_y_rev_ele = list()
for x, y in zip(data_reversed["n_sounds"], data_reversed["coverage"]):
    if x != 1:
        new_x_rev_ele.append(x)
        new_y_rev_ele.append(y)
    else:
        continue

layout = """
ab
"""
fig, ax = plt.subplot_mosaic(mosaic=layout, figsize=(6, 3), sharex=True, sharey=True)

sns.lineplot(x=new_x_clear_azi, y=new_y_clear_azi, err_style="bars", ax=ax["a"])
sns.lineplot(x=new_x_rev_azi, y=new_y_rev_azi, err_style="bars", ax=ax["a"])
ax["a"].set_xlabel("Actual Number Of Sounds")
ax["a"].set_ylabel("Spectro-Temporal Coverage")
ax["a"].legend(["Forward Speech", "Reversed Speech"])
ax["a"].set_ylim([0.5, 1])
ax["a"].set_xlim([1.5, 6.5])
ax["a"].plot(plt.xlim(), plt.ylim(), ls="--", c=".3")

sns.lineplot(x=new_x_clear_ele, y=new_y_clear_ele, err_style="bars", ax=ax["b"])
sns.lineplot(x=new_x_rev_ele, y=new_y_rev_ele, err_style="bars", ax=ax["b"])
ax["b"].set_xlabel("Actual Number Of Sounds")
# ax["a"].set_ylabel("Spectro-Temporal Coverage")
ax["b"].legend(["Forward Speech", "Reversed Speech"])
ax["b"].set_ylim([0.5, 1])
ax["b"].set_xlim([1.5, 6.5])
ax["b"].plot(plt.xlim(), plt.ylim(), ls="--", c=".3")

ax["a"].text(-0.15, 1.05, string.ascii_uppercase[0], transform=ax["a"].transAxes,
             size=20, weight='bold')
ax["b"].text(-0.15, 1.05, string.ascii_uppercase[1], transform=ax["b"].transAxes,
             size=20, weight='bold')

plt.savefig("/home/max/PycharmProjects/BinauralLocalizationCNN/plots/final_spectemp.png",
            dpi=400, bbox_inches="tight")

plt.show()
