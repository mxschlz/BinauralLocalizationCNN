import random

import matplotlib.pyplot as plt
import numpy as np
from stim_util import loc_to_CNNpos
from nnresample import resample
import slab
# import pyroomacoustics as pra
from collections import Sized
from copy import deepcopy
import warnings
from stim_manipulation import change_itd
from pycochleagram import utils as utl


# use Kremar HRTF
KEMAR_HRTF = slab.HRTF.kemar()
# source locations are in hrtf.sources, centered on listener
# default positions
DEFAULT_POSITIONS = [list(range(-90, 91, 10)),    # azimuth
                     10,                          # elevation
                     1.4,                         # distance
                     ]


def pick_hrtf_by_loc(pos_azim=0, pos_elev=0, pos_dist=1.4, interp=False, hrtf_obj=KEMAR_HRTF):
    """
    get the indexes as well as the HRTFs corresponding to the positions encoded in [azim, elev]
    it corresponds to the vertical polar coordinate system of the slab
    currently the distance should not be changed
    # TODO: check the interpolation method
    :param hrtf_obj: hrtf from slab
    :param pos_azim: list-like, azimuth positions w.r.t. listener, degree
    :param pos_elev: list-like, elevation positions, degree
    :param pos_dist: list-like, distance to listener, m
    :param interp: bool, if interpolate the HRTFs not matching the position. currently broken
    :return: hrtfs, indexes, positions
    """
    if not isinstance(pos_azim, Sized):
        pos_azim = [pos_azim]
    if not isinstance(pos_elev, Sized):
        pos_elev = [pos_elev]
    if not isinstance(pos_dist, Sized):
        pos_dist = [pos_dist]

    # generate positions using all combinations of [azim, elev, dist]
    all_pos = []
    for azim in pos_azim:
        for elev in pos_elev:
            for dist in pos_dist:
                all_pos.append([azim, elev, dist])

    # get the hrtf indexes
    hrtf_idx = []
    for pos in all_pos:
        # try find the exact match
        pos_tc = np.array(pos, dtype=hrtf_obj.sources.cartesian.dtype)
        idx = np.where((hrtf_obj.sources.vertical_polar == pos_tc).all(axis=1))[0]
        if idx.size > 0:
            # find match
            hrtf_idx.append(idx[0])
        else:
            # first get nearest neighbor
            cart_allpos = hrtf_obj.sources.cartesian
            cart_target = hrtf_obj._vertical_polar_to_cartesian(np.array(pos).reshape(-1, 3))
            distances = np.sqrt(((cart_target - cart_allpos) ** 2).sum(axis=1))
            idx_nearest = np.argmin(distances)
            hrtf_idx.append(idx_nearest)

    return hrtf_obj, hrtf_idx, all_pos


# default hrtfs
DEFAULT_HRTF_INFO = pick_hrtf_by_loc(*DEFAULT_POSITIONS, )


def augment_from_wav(file_name, meth='hrtf', **kwargs):
    """
    use augment_from_array, but directly works on a .wav file
    :param file_name: path
    :param meth: 'hrtf' or 'room'
    :param kwargs:
    :return:
    """
    stim, stim_sr = utl.wav_to_array(file_name, rescale=None)
    return augment_from_array(stim, stim_sr, meth, **kwargs)


def augment_from_array(sig, sample_rate, meth='hrtf', **kwargs):
    """
    from a single channel audio signal, generate binaural sounds at different locations
    :param sig: np.array, single channel audio signal
    :param sample_rate: Hz
    :param meth: 'hrtf' or 'room'
    :param kwargs:
    :return:
    """
    # input checking
    if 'max_scaling' in kwargs:
        max_scaling = kwargs['max_scaling']
    else:
        max_scaling = 0.1
    if 'hrtfs' in kwargs:
        hrtf_sets = kwargs['hrtfs']
    else:
        hrtf_sets = DEFAULT_HRTF_INFO

    # assuming single channel sound
    if len(sig.shape) == 2:
        if min(sig.shape) > 1:
            warnings.warn('signal contains multiple channels. only take the first channel')
            if np.argmin(sig.shape) == 0:
                sig = sig[0]
            else:
                sig = sig[:, 0]
        else:
            sig = sig.ravel()
    if len(sig.shape) > 2:
        raise ValueError('signal has more than 2 dimensions')
    # ideally we only work with short sounds (<5s), otherwise take too much space
    if len(sig) / sample_rate > 5:
        warnings.warn('signal is more than 5s long. could be RAM consuming')

    # first normalize the signal
    # TODO: does it make sense to do the normalization here?
    sig = max_scaling * utl.rescale_sound(sig, 'normalize')

    # generate localized sound
    if meth == 'hrtf':
        res = simulate_from_hrtf(sig, sample_rate, hrtf_sets, **kwargs)
    elif meth == 'room':
        raise NotImplementedError('Room simulation not implemented yet')
    else:
        raise ValueError('method: {} not known'.format(meth))
    return res


def simulate_from_room():
    pass


def simulate_from_hrtf(sig, sig_sr, HRTFs, target_sr=48000, **kwargs):
    """
    generate binaural sounds from different HRTFs
    :param sig: np.array, single channel audio signal
    :param sig_sr: sample rate of the signal, Hz
    :param HRTFs: output from pick_hrtf_by_loc
    :param target_sr: target sample rate, Hz
    :return:
    """
    # first need to make sure the signal and the filters have the same sample rate as target
    hrtf_sr = HRTFs[0].samplerate
    HRTF_filters = deepcopy(HRTFs[0])
    # check to see if need to resample the HRTFs
    if hrtf_sr != target_sr:
        for idx, hrtf_info in enumerate(HRTF_filters):
            HRTF_filters[idx] = hrtf_info.resample(target_sr)
    if sig_sr != target_sr:
        sig = resample(sig, target_sr, sig_sr, As=75, N=64001)

    selected_hrtfs = HRTFs[1:]
    sig_dicts = []
    # slab functions need slab data types to work
    slab_sig = slab.Sound(sig, samplerate=target_sr)
    for hrtf_idx, pos in zip(*selected_hrtfs):
        bi_sig = HRTF_filters.apply(hrtf_idx, slab_sig)
        # label for current stim
        lb_dict = {'azim': int(pos[0]),
                   'elev': int(pos[1]),
                   'dist': float(pos[2]),
                   'sampling_rate': target_sr,
                   'hrtf_idx': int(hrtf_idx),
                   'cnn_idx': loc_to_CNNpos(pos[0], pos[1])}
        # add additional labels if needed
        if 'extra_lbs' in kwargs:
            lb_dict.update(kwargs['extra_lbs'])
        sig_dicts.append({'sig': bi_sig.data,
                          'label': lb_dict})

    return sig_dicts


def render_stims(orig_stim, pos_azim, pos_elev, hrtf_obj=None, n_reps=1, n_sample=None, **kwargs):
    """
    Renders stimuli with spatial audio effects.

    Args:
        orig_stim (slab.Sound or list): The original sound or a list of sounds to render.
        pos_azim (float): The azimuth position for rendering.
        pos_elev (float): The elevation position for rendering.
        hrtf_obj (object, optional): The Head-Related Transfer Function (HRTF) object. If not provided,
                                    the MIT KEMAR HRTF set is used by default.
        n_reps (int, optional): The number of repetitions for each stimulus in the simulated experiment.
                                Default is 5.
        n_sample (int, optional): The number of original stimuli to sample when rendering a list of sounds.
                                  Default is 1.

    Returns:
        list: A list of dictionaries containing the rendered stimuli and their corresponding labels.
              Each dictionary has the following keys:
              - 'sig': The rendered sound data.
              - 'label': A dictionary containing the label information, including azimuth (azim), ITD, and ILD.

    """

    # Select the appropriate HRTF sets based on the provided or default HRTF object and position
    if not hrtf_obj:
        hrtf_sets = pick_hrtf_by_loc(pos_azim=pos_azim, pos_elev=pos_elev, hrtf_obj=KEMAR_HRTF)
    else:
        hrtf_sets = pick_hrtf_by_loc(pos_azim=pos_azim, pos_elev=pos_elev, hrtf_obj=hrtf_obj)

    stims_rendered = []  # Store the rendered stimuli

    if isinstance(orig_stim, slab.Sound):
        # If the original stimulus is a single sound
        stims_rendered.extend(augment_from_array(orig_stim.data, sample_rate=orig_stim.samplerate, hrtfs=hrtf_sets, **kwargs))
    elif isinstance(orig_stim, list):
        # If the original stimulus is a list of sounds, randomly select a subset for rendering
        if not n_sample:
            n_sample = orig_stim.__len__()
        randsamp = random.sample(orig_stim, n_sample)
        for stim in randsamp:
            stims_rendered.extend(augment_from_array(stim.data, sample_rate=stim.samplerate, hrtfs=hrtf_sets, **kwargs))

    sig_rate = stims_rendered[0]['label']['sampling_rate']

    stims_final = []  # Store the final rendered stimuli

    for rep in range(n_reps):
        # Repeat each stimulus 'n_reps' times to match the original experiment
        for stim_dict in stims_rendered:
            sig = slab.Binaural(stim_dict['sig'], samplerate=sig_rate)
            label_dict = deepcopy(stim_dict['label'])

            # Apply spatial audio effects
            sig = change_itd(sig, azi=label_dict["azim"])
            label_dict['ITD'] = sig.itd()
            label_dict["ILD"] = int(round(sig.ild()))

            # Generate the final dictionary and add it to the list
            stims_final.append({'sig': sig.data, 'label': label_dict})

    return stims_final


if __name__ == "__main__":
    from stim_util import zero_padding
    from show_subbands import show_subbands


    samplerate = 44100  # initial samplerate for CNN
    sound = slab.Sound.pinknoise(0.5, samplerate=samplerate)
    sound = zero_padding(sound, goal_duration=2.1, type="frontback")

    pos_azim = [-90, 0, 90]
    pos_elev = [0]
    stims_rendered = []  # Store the rendered stimuli

    hrtf_sets = pick_hrtf_by_loc(pos_azim=pos_azim, pos_elev=pos_elev, hrtf_obj=KEMAR_HRTF)
    stims_rendered.extend(augment_from_array(sound.data, sample_rate=sound.samplerate, hrtfs=hrtf_sets))


    for i, stm in enumerate(stims_rendered):
        label = stm["label"]
        print(f"Stim {i}: {label}")
        slab.Binaural(stm["sig"], samplerate=samplerate).play()
        show_subbands(slab.Binaural(stm["sig"], samplerate=samplerate))
        plt.show(block=True)



