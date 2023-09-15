# plotting
import matplotlib
matplotlib.use("TkAgg")
import os
from scipy.io import wavfile
import slab
import numpy as np
import matplotlib.pyplot as plt
from CNN_preproc import normalize_binaural_stim, cochleagram_wrapper
from stim_gen import pick_hrtf_by_loc, KEMAR_HRTF
import pickle
from stim_util import *


data_path = "/home/max/labplatform/sound_files"
fn = 'locaaccu_machine_gun_noise.pkl'
file = pickle.load(open(os.path.join(data_path, fn), "rb"))
sig_ori = zero_padding(file[0], goal_duration=2.1)
sig_ori, sf = sig_ori.data, sig_ori.samplerate

# sig = slab.Sound.pinknoise(2.1)
# sig_ori, sf = sig.data, sig.samplerate

hrtf_loc = pick_hrtf_by_loc([45], [10], )
hrtf_loc_idx = hrtf_loc[1]
hrtf_loc_pos = hrtf_loc[2]
sig_slab = slab.Binaural(sig_ori, samplerate=sf)
sig_slab = sig_slab.resample(44100)
# sig_slab.level = 70
sig_bi = KEMAR_HRTF.apply(hrtf_loc_idx[0], sig_slab)


sig_norm = normalize_binaural_stim(sig_bi.data,
                                   sig_bi.samplerate)
subbands = cochleagram_wrapper(sig_norm[0], sig_norm[1], sliced=True)

fig, ax = plt.subplots(3)
t = np.arange(48000) / 48000
ax[0].plot(t, sig_norm[0][0][1000:49000])
ax[0].plot(t, sig_norm[0][1][1000:49000])
ax[0].set_ylabel('Amplitude (arb. unit)')
ax[0].set_title("original signal")
ax[0].legend(['left', 'right'])

ax[1].imshow(subbands[:, :, 0], vmin=0, vmax=0.002, aspect='auto')
ax[1].set_ylabel('Subbands')
ax[1].set_title('left ear')
ax[1].tick_params(axis='x', labelbottom=False)

ax[2].imshow(subbands[:, :, 1], vmin=0, vmax=0.002, aspect='auto')
ax[2].set_ylabel('Subbands')
ax[2].set_title('right ear')
ax[2].set_xticklabels([-0.21, 0, 0.21, 0.42, 0.63, 0.84])
ax[2].set_xlabel('time (s)')
