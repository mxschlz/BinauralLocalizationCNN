from CNN_preproc import process_stims
from tfrecord_gen import create_tfrecord, check_record
import pickle
import slab
import os
import random
import numpy as np

# coords
pos_azim = [0]  # alternative: [-60, -40, -20, 0, 20, 40, 60]
pos_elev = [0, 10, 20, 30, 40, 50, 60]  # alternative: [0, 10, 20, 30, 40, 50, 60]

# load rendered signals
with open(os.path.join(os.getcwd(), f"MSL/numjudge_rendered_stims/pos_azi-{pos_azim}_pos_ele-{pos_elev}_all_talkers_rendered.pkl"), 'rb') as f:
    stims = pickle.load(f)

# simulate MSL experiment from stims
stims_final = dict()
trials = slab.Trialsequence(conditions=[2, 3, 4, 5, 6, 7], n_reps=15).trials
talker_ids = list(stims.keys())
if pos_azim.__len__() == 1:
    dim = "elev"
elif pos_elev.__len__() == 1:
    dim = "azim"
for i, n_sounds in enumerate(trials):  # sounds played in a trial
    sigs_this_trial = list()
    talkers_this_trial = random.sample(talker_ids, n_sounds)
    coords = random.sample(pos_elev, n_sounds)
    for talker, coord in zip(talkers_this_trial, coords):
        sigs = [stims[talker][x]["sig"] for x in range(len(stims[talker]))]
        labels = [stims[talker][x]["label"] for x in range(len(stims[talker]))]
        pool = list()
        for sig, label in zip(sigs, labels):
            if label[dim] == coord:
                pool.append(sig)
    stims_final[f"trial_{i}"] = random.sample(pool, n_sounds)
