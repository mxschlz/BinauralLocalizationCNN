import matplotlib
matplotlib.use("TkAgg")
from stim_gen import *
from CNN_preproc import process_stims
from tfrecord_gen import create_tfrecord, check_record
import pickle
from stim_util import zero_padding
from show_subbands import show_subbands

# DATA GENERATION FOR LOCALIZATION ACCURACY PARADIGM
# get machine run noise: 5 pink noise burst repetitions of 25 ms with 25 ms silence between. zero-padding to 1 second.
# localization accuracy without molds in a 2D grid (-30° to 30° horizontal and vertical, 5° steps)

samplerate = 44100  # initial samplerate for CNN
cochleagram_params = dict(sliced=True, minimum_padding=0.45)
# render the sound
pos_azim = [-30, -15, 0, 15, 30]  # alternative: [-55, -35, -15, 0, 15, 35, 55]
pos_elev = [0]  # alternative: [0, 10, 20, 30, 40, 50, 60]
hrtfs = pick_hrtf_by_loc(pos_azim=pos_azim, pos_elev=pos_elev)


# stim = pickle.load(open("/home/max/labplatform/sound_files/locaaccu_machine_gun_noise.pkl", "rb"))[0]
# stim = stim.repeat(int((2.1 - stim.duration) / stim.duration + 2))
# stim = stim.resample(samplerate)
stim_fp = "/home/max/labplatform/sound_files/locaaccu_machine_gun_noise.pkl"
stim = pickle.load(open(stim_fp, "rb"))[0].ramp()
# resize and resample
stim = stim.resize(0.25).resample(samplerate)
stim.level = 70
stim = zero_padding(stim, goal_duration=2.0, type="frontback")


# load sofa files from CIPIC as ear mold simulation
# sofa_root = os.path.join("tfrecords", "cipic_hrtfs")  # sofa files directory
# cipic_hrtfs = [slab.HRTF(data=os.path.join(sofa_root, x)) for x in os.listdir(sofa_root)]  # hrtfs from CIPIC dataset


stims_final = []
for _ in range(16):
    stims_final.extend(augment_from_array(stim.data, stim.samplerate, hrtfs=hrtfs))


"""
for i, stm in enumerate(stims_final):
    label = stm["label"]
    print(f"Stim {i}: {label}")
    slab.Binaural(stm["sig"], samplerate=samplerate).play()
    show_subbands(slab.Binaural(stm["sig"], samplerate=samplerate))
    plt.show(block=True)
"""

# preprocessing
stims_final = process_stims(stims_final, coch_param=cochleagram_params)
# write tfrecord
rec_file = f'tfrecords/locaaccu_noise_azi_{pos_azim}.tfrecords'
create_tfrecord(stims_final, rec_file)
# check record file
status = check_record(rec_file)

# get babble noise
# render the sound
# get stims
babble_fn = "/home/max/labplatform/sound_files/locaaccu_babble_noise.pkl"
babble = pickle.load(open(babble_fn, "rb"))[0].ramp()
babble = babble.resample(samplerate)
babble.level = 70
babble = zero_padding(babble, goal_duration=2.0, type="frontback")

stims_final = []
for _ in range(16):
    stims_final.extend(augment_from_array(babble.data, babble.samplerate, hrtfs=hrtfs))

# preprocessing
stims_final = process_stims(stims_final, coch_param=cochleagram_params)
# write tfrecord
rec_file = f'tfrecords/locaaccu_babble_azi_{pos_azim}.tfrecords'
create_tfrecord(stims_final, rec_file)
# check record file
status = check_record(rec_file)
