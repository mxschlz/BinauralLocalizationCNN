import itertools
import pickle
import random

import numpy as np
import slab

from CNN_preproc import transform_stims_to_cochleagrams
from legacy.CNN_util import get_dataset_partitions
from legacy.stim_gen import render_stims
from legacy.stim_util import zero_padding
from legacy.tfrecord_gen import create_tfrecord, check_record

# PIPELINE FOR NUMJUDGE TFRECORD GENERATION
# Render each talker for a given spatial position (azi, ele) in a trial separately.
# Afterwards, add everything together.
samplerate = 44100  # initial samplerate for CNN
goal_duration = 2.0  # CNN processing goal duration
goal_level = 70  # I think this is the level as proposed in the francl paper

# simulate experiment sequence by rendering and adding sounds to create a complex multi-source environment
# render the sound
pos_azim = [-60, -40, -20, 0, 20, 40, 60]  # alternative: [-60, -40, -20, 0, 20, 40, 60]
pos_elev = [0, 10, 20, 30, 40, 50, 60]  # alternative: [0, 10, 20, 30, 40, 50, 60]
stim_n_reps = 20  # number of stimulus repetitions
exp_n_reps = 1  # number of condition repetitions in the trial sequence
n_countries = 13
conditions = [1, 2, 3, 4, 5, 6]
cochleagram_params = dict(sliced=True, minimum_padding=0.45)

# get stims from original experiment
talkers_clear = pickle.load(open("/home/max/labplatform/sound_files/numjudge_talker_files_clear.pkl", "rb"))
log = slab.ResultsFile(folder="logfiles", subject="log_train_test")

# gather all available stimuli
stimlist_clear = dict()

for talker in list(talkers_clear.keys()):
    stimlist_clear[talker] = list()

    for stim in talkers_clear[talker]:
        # stim = stim.resample(samplerate)
        stimlist_clear[talker].append(stim)

sequence = slab.Trialsequence(conditions=conditions, n_reps=exp_n_reps)
talker_ids = list(stimlist_clear.keys())
final_stims_ele = list()
final_stims_azi = list()

# iterate once through all possible numbers of sources
for i, _ in enumerate(sequence):
    # AZIMUTH
    # choose number of source
    n_sounds = sequence.this_trial
    print(f"Source configuration number: {n_sounds}")
    log.write(n_sounds, "n_sounds")
    # get all possible coordinate combinations for this particular number of sounds
    all_source_combinations_azi = list(itertools.combinations(pos_azim, n_sounds))
    for i, comb in enumerate(all_source_combinations_azi):
        print(f"Computation {i} / {len(all_source_combinations_azi)} for azimuth")
        for rep in range(stim_n_reps):
            print(f"Rep {rep} / {stim_n_reps}")
            sound = slab.Binaural(data=np.zeros(samplerate), samplerate=samplerate)
            talkers_samp = random.sample(talker_ids, n_sounds)
            country_idxs = random.sample(range(n_countries), n_sounds)
            binary_label = np.zeros(504, dtype=np.int64)
            log.write(talkers_samp, "signals_sample_azi")
            log.write(comb, "speakers_sample_azi")
            log.write(country_idxs, "country_idxs_azi")
            for az, talker, country_idx in zip(comb, talkers_samp, country_idxs):
                print(f"Adding {az}, {talker}, {country_idx} ... ")
                stim = render_stims(stimlist_clear[talker][country_idx], pos_azim=az, pos_elev=0)
                binary_label[stim[0]["label"]["cnn_idx"]] = 1
                sound += slab.Binaural(data=stim[0]["sig"], samplerate=samplerate).resample(samplerate).resize(
                    len(sound))
            sound = zero_padding(sound, type="front", goal_duration=goal_duration)
            sound.level = [goal_level, goal_level]
            final_stims_azi.append({"sig": sound.data, "label": {"n_sounds": n_sounds, "sampling_rate": samplerate,
                                                                 "hrtf_idx": 0, "binary_label": binary_label}})
            print(f"Azimuth sample computed. Adding to final list ...")
            # show_subbands(sound)
            # sound.play()
    # ELEVATION
    all_source_combinations_ele = list(itertools.combinations(pos_elev, n_sounds))
    for comb in all_source_combinations_ele:
        print(f"Computation {i} / {len(all_source_combinations_ele)} for elevation")
        for rep in range(stim_n_reps):
            print(f"Rep {rep} / {stim_n_reps}")
            sound = slab.Binaural(data=np.zeros(samplerate), samplerate=samplerate)
            talkers_samp = random.sample(talker_ids, n_sounds)
            country_idxs = random.sample(range(n_countries), n_sounds)
            binary_label = np.zeros(504, dtype=np.int64)
            log.write(talkers_samp, "signals_sample_ele")
            log.write(comb, "speakers_sample_ele")
            log.write(country_idxs, "country_idxs_ele")
            for el, talker, country_idx in zip(comb, talkers_samp, country_idxs):
                print(f"Adding {el}, {talker}, {country_idx} ... ")
                stim = render_stims(stimlist_clear[talker][country_idx], pos_azim=0, pos_elev=el)
                binary_label[stim[0]["label"]["cnn_idx"]] = 1
                sound += slab.Binaural(data=stim[0]["sig"], samplerate=samplerate).resample(samplerate).resize(
                    len(sound))
            sound = zero_padding(sound, type="front", goal_duration=goal_duration)
            sound.level = [goal_level, goal_level]
            final_stims_ele.append({"sig": sound.data, "label": {"n_sounds": n_sounds, "sampling_rate": samplerate,
                                                                 "hrtf_idx": 0, "binary_label": binary_label}})
            print(f"Elevation sample computed. Adding to final list ...")
            # show_subbands(sound)
            # sound.play()
    print(f"Trial number {i + 1}/{sequence.n_trials} finished. Continuing ...")
print("Data generation done!")

# divide into train and test set
for i, stim_dset in enumerate([final_stims_azi, final_stims_ele]):
    plane = "azi" if i == 0 else "ele"
    coords = pos_azim if plane == "azi" else pos_elev
    train, test = get_dataset_partitions(stim_dset, train_split=0.8, test_split=0.2, shuffle=True)
    for name, ds in zip(["train", "test"], [train, test]):
        # preprocessing
        ds_final = transform_stims_to_cochleagrams(ds, coch_param=cochleagram_params)

        # write tfrecord
        rec_file_ds = f'numjudge_talkers_clear_{name}_{plane}_{coords}.tfrecords'
        create_tfrecord(ds_final, rec_file_ds)

        # check record file
        print(f"{plane} TFrecords {name} set successful: ", check_record(rec_file_ds))
