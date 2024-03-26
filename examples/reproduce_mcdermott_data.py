import slab
import os
from copy import deepcopy

from stim_gen import augment_from_array, pick_hrtf_by_loc, render_stims
from CNN_preproc import process_stims
from tfrecord_gen import create_tfrecord, check_record

# NOTE: cochleagram takes a lot of RAM; large datasets, e.g. ITD/ILD takes ~10 GB ram
# run them separately if you cannot run them in the same python session due to the RAM issue
# for all the slab generation
sample_rate = 44100
# 16 white noise
sigs = []
for _ in range(16):
    sigs.append(slab.Sound.whitenoise(samplerate=sample_rate, duration=2.1).ramp())
# binauralize the sounds using stim_gen
stims_bi = []
for sig in sigs:
    stims_bi.extend(augment_from_array(sig.data, sig.samplerate))

# preprocessing
stims_bi = process_stims(stims_bi)

# write tfrecord
rec_path = os.path.join('tfrecords', 'mcdermott')
rec_file = os.path.join(rec_path, 'broadband_noise_azimuth.tfrecords')
create_tfrecord(stims_bi, rec_file)
# check record file
status = check_record(rec_file)

# same stimulus, filtered with HRTFs for elevation
elevations = [0, 10, 20, 30, 40, 50, 60]
hrtfs = pick_hrtf_by_loc(pos_elev=elevations)
sigs = []
for _ in range(16):
    sigs.append(slab.Sound.whitenoise(samplerate=sample_rate, duration=2.1).ramp())
# binauralize the sounds using stim_gen
stims_bi = []
for sig in sigs:
    stims_bi.extend(augment_from_array(sig.data, sig.samplerate, hrtfs=hrtfs))
# preprocessing
stims_bi = process_stims(stims_bi)

# write tfrecord
rec_path = os.path.join('tfrecords', 'mcdermott')
rec_file = os.path.join(rec_path, 'broadband_noise_elevation.tfrecords')
create_tfrecord(stims_bi, rec_file)
# check record file
status = check_record(rec_file)

# data generation for figure 2
# filter the noise to get narrow-band noise
sigs = []
for _ in range(4):
    sigs.append(slab.Sound.whitenoise(samplerate=sample_rate, duration=2.1))
filt_low = slab.Filter.band('bp', frequency=(400, 2000),
                            samplerate=sample_rate, length=2048)
filt_high = slab.Filter.band('bp', frequency=(3900, 16100),
                             samplerate=44100, length=2048)
sigs_low = []
sigs_high = []
for sig_ori in sigs:
    sigs_low.append(filt_low.apply(sig_ori))
    sigs_high.append(filt_high.apply(sig_ori))

# binauralize the sounds using stim_gen
hrtf_sets = pick_hrtf_by_loc(pos_azim=list(range(-180, 171, 20)),
                             pos_elev=20)
stims_low_bi = []
for sig in sigs_low:
    stims_low_bi.extend(augment_from_array(sig.data, sig.samplerate, hrtfs=hrtf_sets,
                                           extra_lbs={'center_freq': 1000, 'bandwidth': 2}))
stims_high_bi = []
for sig in sigs_high:
    stims_high_bi.extend(augment_from_array(sig.data, sig.samplerate, hrtfs=hrtf_sets,
                                            extra_lbs={'center_freq': 8000, 'bandwidth': 2}))

# manipulate ITD/ILD
# for both ITD/ILD, negative values shift the sound to the left
# only need to modulate them separately
ITD_bias = [-600, -300, 0, 300, 600]  # in micro seconds
sig_rate = stims_low_bi[0]['label']['sampling_rate']
ILD_bias = [-20, -10, 0, 10, 20]  # in dB
stims_high_final = []
for itd in ITD_bias:
    for sig_dict in stims_high_bi:
        # convert to slab.Binaural
        sig = slab.Binaural(sig_dict['sig'], samplerate=sig_rate)
        label_dict = deepcopy(sig_dict['label'])
        # change ITD
        itd_ns = int(itd * sig_rate / 1e6)
        sig = sig.itd(itd_ns)
        # update label dict
        label_dict['ITD'] = itd
        label_dict['ILD'] = 0
        # generate final dict list
        stims_high_final.append({'sig': sig.data,
                                 'label': label_dict})

for ild in ILD_bias:
    for sig_dict in stims_high_bi:
        # convert to slab.Binaural
        sig = slab.Binaural(sig_dict['sig'], samplerate=sig_rate)
        label_dict = deepcopy(sig_dict['label'])
        # change ILD
        sig = sig.ild(sig.ild() + ild)
        # update label dict
        label_dict['ITD'] = 0
        label_dict['ILD'] = ild
        # generate final dict list
        stims_high_final.append({'sig': sig.data,
                                 'label': label_dict})

# preprocessing and write
stims_high_final = process_stims(stims_high_final)
# write tfrecord
rec_path = os.path.join('tfrecords', 'mcdermott')
rec_file = os.path.join(rec_path, 'noise_high_ITDILD.tfrecords')
create_tfrecord(stims_high_final, rec_file)
# check record file
status = check_record(rec_file)

stims_low_final = []
for itd in ITD_bias:
    for sig_dict in stims_low_bi:
        # convert to slab.Binaural
        sig = slab.Binaural(sig_dict['sig'], samplerate=sig_rate)
        label_dict = deepcopy(sig_dict['label'])
        # change ITD
        itd_ns = int(itd * sig_rate / 1e6)
        sig = sig.itd(itd_ns)
        # update label dict
        label_dict['ITD'] = itd
        label_dict['ILD'] = 0
        # generate final dict list
        stims_low_final.append({'sig': sig.data,
                                'label': label_dict})

for ild in ILD_bias:
    for sig_dict in stims_low_bi:
        # convert to slab.Binaural
        sig = slab.Binaural(sig_dict['sig'], samplerate=sig_rate)
        label_dict = deepcopy(sig_dict['label'])
        # change ILD
        sig = sig.ild(sig.ild() + ild)
        # update label dict
        label_dict['ITD'] = 0
        label_dict['ILD'] = ild
        # generate final dict list
        stims_low_final.append({'sig': sig.data,
                                'label': label_dict})
# process and write
stims_low_final = process_stims(stims_low_final)
# write tfrecord
rec_path = os.path.join('tfrecords', 'mcdermott')
rec_file = os.path.join(rec_path, 'noise_low_ITDILD.tfrecords')
create_tfrecord(stims_low_final, rec_file)
# check record file
status = check_record(rec_file)

# TODO: cannot generate band limited noise due to RAM exhaustion
# band-limited noise
center_freq = [1000, 2000, 4000]
bandwidth = [0, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1]
# filter the noise to get narrow-band noise
sigs = []
for _ in range(3):
    sigs.append(slab.Sound.whitenoise(samplerate=sample_rate, duration=2.1))
sigs_bp_bi = []
for cf in center_freq:
    for bd in bandwidth:
        sigs_temp = []
        if bd == 0:
            sigs_temp.append(slab.Sound.tone(frequency=cf, samplerate=sample_rate, duration=2.1))
        else:
            for sig in sigs:
                # calculate frequency limits from cf and bd
                fl = [cf / 2 ** (bd / 2), cf * 2 ** (bd / 2)]
                filt = slab.Filter.band('bp', frequency=tuple(fl),
                                        samplerate=sample_rate, length=2048)
                sigs_temp.append(filt.apply(sig))
        # binauralize
        for st in sigs_temp:
            sigs_bi = augment_from_array(st.data, st.samplerate)
            for stb in sigs_bi:
                label_dict = deepcopy(stb['label'])
                label_dict.update({
                    'center_freq': int(cf),
                    'bandwidth': float(bd),
                })
                stb['label'] = label_dict
                sigs_bp_bi.append(stb)

# preprocessing
sigs_bp_bi = process_stims(sigs_bp_bi)
# write tfrecord
rec_path = os.path.join('tfrecords', 'mcdermott')
rec_file = os.path.join(rec_path, 'noise_bandwidth.tfrecords')
create_tfrecord(sigs_bp_bi, rec_file)
# check record file
status = check_record(rec_file)
