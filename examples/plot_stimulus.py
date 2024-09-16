import os

import matplotlib.pyplot as plt
import numpy as np
import slab

from stim_gen import augment_from_wav, augment_from_array, pick_hrtf_by_loc, KEMAR_HRTF

"""
Takes a sound and augments it with HRTFs from KEMAR, then plots stuff. Unsure what exactly.
Might be useful for my own plots...
No usages.
"""

data_path = os.path.join("examples", 'stimuli')
wav_ori = 'test.wav'
wf = os.path.join(data_path, wav_ori)
sound = slab.Sound.read(wf)
sig_ori, sf = sound.data, sound.samplerate
# just need a single channel for the room simulation
sig_ori = sig_ori[:, 0] / np.max(sig_ori)

# sound with hrtf
bsig_hrtf = augment_from_array(sig_ori, sf)  # unused
# or directly from .wav file
bsig_hrtf = augment_from_wav(wf)  # unused

# plotting
# pick hrtf at -45 and 45, both at elevation of 10
hrtf_loc = pick_hrtf_by_loc([-45, 45], [10], )
hrtf_loc_idx = hrtf_loc[1]
hrtf_loc_pos = hrtf_loc[2]
fig, ax = plt.subplots(3, 2, )
fig.set_size_inches(7.5, 6)
# original sound
t = np.arange(len(sig_ori)) / sf
ax[0][0].plot(t, sig_ori)
ax[0][0].set_xlabel('time (s)')
ax[0][0].set_ylabel('Amplitude (Arb. unit)')
ax[0][0].set_title('Original sound')
ax[0][1].set_axis_off()
sig_slab = slab.Sound(sig_ori, samplerate=sf)
for ct, loc in enumerate(hrtf_loc_idx):
    filt = KEMAR_HRTF.data[loc]
    w, h = filt.tf(show=False)
    ax[ct + 1][0].plot(w, h)
    ax[ct + 1][0].set_ylabel("Amplitude (dB)")
    if ct == 0:
        ax[ct + 1][0].set_title('Frequency Response')
        ax[ct + 1][0].legend(['Left', 'Right'])
    elif ct == 1:
        ax[ct + 1][0].set_xlabel('Frequency (Hz)')

    if filt.samplerate != sf:
        filt = filt.resample(sf)
    sig_filt = filt.apply(sig_slab)
    if hrtf_loc_pos[ct][0] < 0:
        ax[ct + 1][1].plot(t, sig_filt.data[:, 1], color='#ff7f0e')
        ax[ct + 1][1].plot(t, sig_filt.data[:, 0], color='#1f77b4')
    else:
        ax[ct + 1][1].plot(t, sig_filt.data[:, 0])
        ax[ct + 1][1].plot(t, sig_filt.data[:, 1])
    if ct == 0:
        ax[ct + 1][1].set_title('Binaural sound')
    else:
        ax[ct + 1][1].set_xlabel('time (s)')
    ax[ct + 1][1].set_ylabel('Amplitude')

plt.tight_layout()
plt.show()
