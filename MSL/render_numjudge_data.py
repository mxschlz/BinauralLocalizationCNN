import os
from stim_gen import *
import pickle

# DATA GENERATION FOR NUMEROSITY JUDGEMENT PARADIGM
# Render each talker for a given spatial position (azi, ele) in a trial separately. Afterwards, add everything together.
samplerate = 44100  # initial samplerate for CNN

# get stims from original experiment
talkers_clear = pickle.load(open("/home/max/labplatform/sound_files/numjudge_talker_files_clear.pkl", "rb"))
talkers_reversed = pickle.load(open("/home/max/labplatform/sound_files/numjudge_talker_files_reversed.pkl", "rb"))

# simulate experiment sequence by rendering and adding sounds to create a complex multi-source environment
# render the sound
pos_azim = [0]  # alternative: [-60, -40, -20, 0, 20, 40, 60]
pos_elev = [0, 10, 20, 30, 40, 50, 60]  # alternative: [0, 10, 20, 30, 40, 50, 60]

# gather all available stimuli
stimlist_clear = dict()
for talker in list(talkers_clear.keys()):
    stimlist_clear[talker] = list()
    for stim in talkers_clear[talker]:
        stimlist_clear[talker].append(stim)

# render stimuli
stims_clear = dict()
for talker in list(stimlist_clear.keys()):
    stims_clear[talker] = render_stims(stimlist_clear[talker], pos_azim=pos_azim, pos_elev=pos_elev, hrtf_obj=KEMAR_HRTF, n_reps=1,
                                       extra_lbs={"talker_id":int(talker)})

# save rendered stims
with open(os.path.join(os.getcwd(), f"MSL/numjudge_rendered_stims/pos_azi-{pos_azim}_pos_ele-{pos_elev}_all_talkers_clear_rendered.pkl"), "wb") as file:
    pickle.dump(stims_clear, file)

# reversed stimuli
stimlist_reversed = dict()
for talker in list(talkers_clear.keys()):
    stimlist_reversed[talker] = list()
    for stim in talkers_reversed[talker]:
        stimlist_reversed[talker].append(stim)

# render stimuli
stims_reversed = dict()
for talker in list(stimlist_reversed.keys()):
    stims_reversed[talker] = render_stims(stimlist_reversed[talker], pos_azim=pos_azim, pos_elev=pos_elev, hrtf_obj=KEMAR_HRTF, n_reps=1,
                                          extra_lbs={"talker_id":int(talker)})

# save rendered stims
with open(os.path.join(os.getcwd(), f"MSL/numjudge_rendered_stims/pos_azi-{pos_azim}_pos_ele-{pos_elev}_all_talkers_reversed_rendered.pkl"), "wb") as file:
    pickle.dump(stims_reversed, file)
