import os
from stim_gen import *
import pickle
from CNN_preproc import process_stims
from stim_util import zero_padding
from tfrecord_gen import create_tfrecord, check_record
from CNN_util import get_dataset_partitions
import slab
# from show_subbands import show_subbands


# PIPELINE FOR NUMJUDGE TFRECORD GENERATION
# Render each talker for a given spatial position (azi, ele) in a trial separately.
# Afterwards, add everything together.
samplerate = 44100  # initial samplerate for CNN
goal_duration = 2.0  # CNN processing goal duration

# simulate experiment sequence by rendering and adding sounds to create a complex multi-source environment
# render the sound
pos_azim = [-60, -40, -20, 0, 20, 40, 60]  # alternative: [-60, -40, -20, 0, 20, 40, 60]
pos_elev = [0, 10, 20, 30, 40, 50, 60]  # alternative: [0, 10, 20, 30, 40, 50, 60]
stim_n_reps = 1  # number of stimulus repetitions
exp_n_reps = 619  # number of condition repetitions in the trial sequence
n_countries = 13
cochleagram_params = dict(sliced=True, minimum_padding=0.45)

# get stims from original experiment
talkers_clear = pickle.load(open("/home/max/labplatform/sound_files/numjudge_talker_files_clear.pkl", "rb"))
log = slab.ResultsFile(folder="logfiles", subject="log_train_test")

# gather all available stimuli
stimlist_clear = dict()

for talker in list(talkers_clear.keys()):
    stimlist_clear[talker] = list()

    for stim in talkers_clear[talker]:
        stimlist_clear[talker].append(stim)

sequence = slab.Trialsequence(conditions=[2, 3, 4, 5, 6], n_reps=exp_n_reps)
talker_ids = list(stimlist_clear.keys())
final_stims_ele = list()
final_stims_azi = list()

for i, _ in enumerate(sequence):

    sound = slab.Binaural(data=np.zeros(samplerate), samplerate=samplerate)
    n_sounds = sequence.this_trial
    talkers_this_trial = random.sample(talker_ids, n_sounds)
    azi = random.sample(pos_azim, n_sounds)
    country_idxs = random.sample(range(n_countries), n_sounds)
    log.write(n_sounds, "n_sounds")
    log.write(talkers_this_trial, "signals_sample")
    log.write(azi, "speakers_sample_azimuth")
    log.write(country_idxs, "country_idxs")

    for az, talker, country_idx in zip(azi, talkers_this_trial, country_idxs):
        stim = render_stims(stimlist_clear[talker][country_idx], pos_azim=az, pos_elev=0, n_reps=stim_n_reps)
        sound += slab.Binaural(data=stim[0]["sig"], samplerate=samplerate).resample(samplerate).resize(len(sound))
    sound = zero_padding(sound, type="front", goal_duration=goal_duration)
    # show_subbands(sound)
    final_stims_azi.append({"sig": sound.data, "label": {"n_sounds": n_sounds, "sampling_rate": samplerate,
                                                         "hrtf_idx": 0}})
    # sound.play()
    # print("n_sounds", n_sounds)

    sound = slab.Binaural(data=np.zeros(samplerate), samplerate=samplerate)
    ele = random.sample(pos_elev, n_sounds)
    log.write(ele, "speakers_sample_elevation")

    for el, talker, country_idx in zip(ele, talkers_this_trial, country_idxs):
        stim = render_stims(stimlist_clear[talker][country_idx], pos_azim=0, pos_elev=el, n_reps=stim_n_reps)
        sound += slab.Binaural(data=stim[0]["sig"], samplerate=samplerate).resample(samplerate).resize(len(sound))
    sound = zero_padding(sound, type="front", goal_duration=goal_duration)

    final_stims_ele.append({"sig": sound.data, "label": {"n_sounds": n_sounds, "sampling_rate": samplerate,
                                                         "hrtf_idx": 0}})
    # show_subbands(sound)
    # sound.play()
    # print("n_sounds", n_sounds)
    print(f"Trial number {i+1}/{sequence.n_trials} finished. Continuing ...")

# divide into train and test set
for i, stim_dset in enumerate([final_stims_azi, final_stims_ele]):
    plane = "azi" if i == 0 else "ele"
    coords = pos_azim if plane == "azi" else pos_elev
    train, test = get_dataset_partitions(stim_dset, train_split=0.8, test_split=0.2, shuffle=True)
    # preprocessing
    train_final = process_stims(train, coch_param=cochleagram_params)
    test_final = process_stims(test, coch_param=cochleagram_params)

    # write tfrecord
    rec_file_train = f'numjudge_full_set_talkers_clear_train_{plane}_{coords}.tfrecords'
    rec_file_test = f'numjudge_full_set_talkers_clear_test_{plane}_{coords}.tfrecords'
    create_tfrecord(train_final, rec_file_train)
    create_tfrecord(test_final, rec_file_test)

    # check record file
    print(f"{plane} TFrecords training set successful: ", check_record(rec_file_train))
    print(f"{plane} TFrecords testing set successful: ", check_record(rec_file_test))


