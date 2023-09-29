import os
from stim_gen import *
import pickle
from CNN_preproc import process_stims
from stim_util import zero_padding
from tfrecord_gen import create_tfrecord, check_record
# from show_subbands import show_subbands

# PIPELINE FOR NUMJUDGE TFRECORD GENERATION
# Render each talker for a given spatial position (azi, ele) in a trial separately. Afterwards, add everything together.
samplerate = 44100  # initial samplerate for CNN
goal_duration = 2.1  # CNN processing goal duration

# get stims from original experiment
talkers_clear = pickle.load(open("/home/max/labplatform/sound_files/numjudge_talker_files_clear.pkl", "rb"))

# simulate experiment sequence by rendering and adding sounds to create a complex multi-source environment
# render the sound
pos_azim = [-60, -40, -20, 0, 20, 40, 60]  # alternative: [-60, -40, -20, 0, 20, 40, 60]
pos_elev = [0, 10, 20, 30, 40, 50, 60]  # alternative: [0, 10, 20, 30, 40, 50, 60]

# gather all available stimuli
stimlist_clear = dict()

for talker in list(talkers_clear.keys()):
    stimlist_clear[talker] = list()

    for stim in talkers_clear[talker]:
        stimlist_clear[talker].append(stim)


sequence = slab.Trialsequence(conditions=[2, 3, 4, 5, 6], n_reps=1)
talker_ids = list(stimlist_clear.keys())
final_stims = list()

for i, _ in enumerate(sequence):

    sound = slab.Binaural(data=np.zeros(samplerate), samplerate=samplerate)
    n_sounds = sequence.this_trial
    talkers_this_trial = random.sample(talker_ids, n_sounds)
    azi = random.sample(pos_azim, n_sounds)

    for az, talker in zip(azi, talkers_this_trial):
        stim = render_stims(random.choice(stimlist_clear[talker]), pos_azim=az, pos_elev=0, n_reps=1)
        sound += slab.Binaural(data=stim[0]["sig"], samplerate=samplerate).resample(samplerate).resize(len(sound))
    sound = zero_padding(sound, goal_duration=goal_duration)
    # show_subbands(sound)
    final_stims.append({"sig": sound.data, "label": {"n_sounds": n_sounds, "sampling_rate": samplerate,
                                                     "hrtf_idx": 0}})
    # sound.play()
    # print("n_sounds", n_sounds)

    sound = slab.Binaural(data=np.zeros(samplerate), samplerate=samplerate)
    ele = random.sample(pos_elev, n_sounds)

    for el, talker in zip(ele, talkers_this_trial):
        stim = render_stims(random.choice(stimlist_clear[talker]), pos_azim=0, pos_elev=el, n_reps=1)
        sound += slab.Binaural(data=stim[0]["sig"], samplerate=samplerate).resample(samplerate).resize(len(sound))
    sound = zero_padding(sound, goal_duration=goal_duration)

    final_stims.append({"sig": sound.data, "label": {"n_sounds": n_sounds, "sampling_rate": samplerate,
                                                     "hrtf_idx": 0}})
    # show_subbands(sound)
    # sound.play()
    # print("n_sounds", n_sounds)

# preprocessing
stims_final = process_stims(final_stims)
# write tfrecord
rec_path = os.path.join("tfrecords/msl")
rec_file = os.path.join(rec_path, f'numjudge_full_set_talkers_clear_train.tfrecords')
create_tfrecord(stims_final, rec_file)
# check record file
status = check_record(rec_file)



