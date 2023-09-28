import matplotlib
matplotlib.use("TkAgg")
import slab
import os
from stim_gen import *
from CNN_preproc import process_stims
from tfrecord_gen import create_tfrecord, check_record
import pickle

# DATA GENERATION FOR LOCALIZATION ACCURACY PARADIGM
# get machine run noise: 5 pink noise burst repetitions of 25 ms with 25 ms silence between. zero-padding to 1 second.
# localization accuracy without molds in a 2D grid (-30° to 30° horizontal and vertical, 5° steps)

samplerate = 44100  # initial samplerate for CNN

# stim = pickle.load(open("/home/max/labplatform/sound_files/locaaccu_machine_gun_noise.pkl", "rb"))[0]
# stim = stim.repeat(int((2.1 - stim.duration) / stim.duration + 2))
# stim = stim.resample(samplerate)
stim = slab.Sound.pinknoise(2.1, samplerate=samplerate)


# load sofa files from CIPIC as ear mold simulation
# sofa_root = os.path.join("tfrecords", "cipic_hrtfs")  # sofa files directory
# cipic_hrtfs = [slab.HRTF(data=os.path.join(sofa_root, x)) for x in os.listdir(sofa_root)]  # hrtfs from CIPIC dataset

# render the sound
pos_azim = [0]  # alternative: [-52.5, -35, -17.5, 0, 17.5, 35, 52.5]
pos_elev = [0, 10, 20, 30, 40]  # alternative: [0, 10, 20, 30, 40]

stims_final = render_stims(orig_stim=stim, pos_azim=pos_azim, pos_elev=pos_elev, hrtf_obj=KEMAR_HRTF, n_reps=5)

# preprocessing
stims_final = process_stims(stims_final)
# write tfrecord
rec_path = os.path.join("tfrecords/msl/cnn")
rec_file = os.path.join(rec_path, 'locaaccu_noise_v.tfrecords')
create_tfrecord(stims_final, rec_file)
# check record file
status = check_record(rec_file)

# get babble noise
# render the sound
# get stims
babble_fn = "/home/max/labplatform/sound_files/locaaccu_babble_noise.pkl"
babble = pickle.load(open(babble_fn, "rb"))
babble = [x.repeat(4) for x in babble]  # extend duration to > 2s
babble = [x.resample(samplerate) for x in babble]

stims_final = render_stims(orig_stim=babble, pos_azim=pos_azim, pos_elev=pos_elev, hrtf_obj=KEMAR_HRTF, n_sample=1, n_reps=5)

# preprocessing
stims_final = process_stims(stims_final)
# write tfrecord
rec_path = os.path.join("tfrecords/msl/cnn")
rec_file = os.path.join(rec_path, 'locaaccu_babble_v.tfrecords')
create_tfrecord(stims_final, rec_file)
# check record file
status = check_record(rec_file)
