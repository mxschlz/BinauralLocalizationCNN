from analysis_and_plotting.misc import spectemp_coverage
import slab
import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="white", palette="viridis")
plt.ion()

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

new_x_clear = list()
new_y_clear = list()
for x, y in zip(data_clear["n_sounds"], data_clear["coverage"]):
    if x != 1:
        new_x_clear.append(x)
        new_y_clear.append(y)
    else:
        continue

new_x_rev = list()
new_y_rev = list()
for x, y in zip(data_reversed["n_sounds"], data_reversed["coverage"]):
    if x != 1:
        new_x_rev.append(x)
        new_y_rev.append(y)
    else:
        continue

sns.lineplot(x=new_x_clear, y=new_y_clear, err_style="bars")
sns.lineplot(x=new_x_rev, y=new_y_rev, err_style="bars")
plt.xlabel("Actual Number Of Sounds")
plt.ylabel("Spectro-Temporal Coverage")
plt.legend(["Forward Speech", "Reversed Speech"])
plt.ylim([0.5, 1])
plt.xlim([1.5, 6.5])
plt.plot(plt.xlim(), plt.ylim(), ls="--", c=".3")
plt.show()
