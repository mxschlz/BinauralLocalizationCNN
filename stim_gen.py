import random
import warnings

import numpy as np
from typing import Iterable

from matplotlib import pyplot as plt

from generate_dataset import main
from stim_util import loc_to_CNNpos
from nnresample import resample
import slab
# import pyroomacoustics as pra
from copy import deepcopy
from stim_manipulation import change_itd
from pycochleagram import utils
import itertools

# use Kemar HRTF
KEMAR_HRTF = slab.HRTF.kemar()
KEMAR_SAMPLERATE = KEMAR_HRTF.samplerate
# source locations are in hrtf.sources, centered on listener

# Deprecated: Not needed anymore, also slab.HRTF has a method to pick the HRTF by location
def pick_hrtf_by_loc(pos_azim=0, pos_elev=0, pos_dist=1.4, interpolate=False, hrtf_obj: slab.HRTF = KEMAR_HRTF):
    """
    Returns the indices as well as the HRTFs corresponding to the positions encoded in [pos_azim, pos_elev, pos_dist]
    It corresponds to slab's vertical polar coordinate system.
    Currently the distance should not be changed. (why?)
    # TODO: Implement interpolation (but I think not here, rather in slab to keep everything in the HRTF class)
    :param hrtf_obj: HRTF from slab
    :param pos_azim: int or Iterable, azimuth positions w.r.t. listener, degree
    :param pos_elev: int or Iterable, elevation positions, degree
    :param pos_dist: int or Iterable, distance to listener, m
    :param interpolate: bool, if True, interpolate the HRTFs not matching the position. not implemented
    :return: HRTF object passed through, indices, positions
    """

    ## If the input is not an iterable, make it an iterable (could use further checks but works for now)
    if not isinstance(pos_azim, Iterable):
        pos_azim = [pos_azim]
    if not isinstance(pos_elev, Iterable):
        pos_elev = [pos_elev]
    if not isinstance(pos_dist, Iterable):
        pos_dist = [pos_dist]

    # generate positions using all combinations of (azim, elev, dist)
    positions = [list(itertools.product(pos_azim, pos_elev, pos_dist))]

    # get the hrtf indices. What are they?? ->
    hrtf_indices = []
    for position in positions:
        # try to find an exact match

        # Shouldn't next line have vertical_polar instead of cartesian?
        pos_tc = np.array(position, dtype=hrtf_obj.sources.cartesian.dtype)  # produces an array with 1 element
        # -> np.array([[90, 0, 1.4]])

        # hrtf_obj.sources is a namedtuple ('cartesian vertical_polar interaural_polar') of lists of tuples (coordinates)
        # hrtf_obj.sources.vertical_polar is a list of coordinates
        # -> np.array([[-90, 0, 1.4], [-80, 0, 1.4], ..., [90, 0, 1.4]])

        # == on two np arrays returns an array of booleans, broadcasting the smaller array to the larger one
        # if all elements are true, then azim, elev, and dist are the same and for this pos we have an HRTF
        # .all works on axis 1, so it checks if all elements in the row are True
        # -> returns an array of booleans, one for each row with a True where a matching HRTF coordinate was found
        # We're just interested in the index (not the value) as it's the index in the HRTF object
        # nonzero returns tuple of arrays, one for each dimension (here 1), containing the indices of the non-zero elements
        # -> [0] to get the array of indices of the first and only dimension

        idx = np.nonzero((hrtf_obj.sources.vertical_polar == pos_tc).all(axis=1))[0]  # unclear...
        if idx.size > 0:  # if found
            hrtf_indices.append(idx[0])  # append the index
        else:
            # first get nearest neighbour
            cart_allpos = hrtf_obj.sources.cartesian
            cart_target = hrtf_obj._vertical_polar_to_cartesian(np.array(position).reshape(-1, 3))
            distances = np.sqrt(((cart_target - cart_allpos) ** 2).sum(axis=1))
            idx_nearest = np.argmin(distances)
            hrtf_indices.append(idx_nearest)

    return hrtf_obj, hrtf_indices, positions


# default hrtfs
DEFAULT_HRTF_INFO = pick_hrtf_by_loc(*DEFAULT_POSITIONS, )
DEFAULT_HRTF_INFO = None


def augment_from_wav(file_name, meth='hrtf', **kwargs):
    """
    Wraps augment_from_array to use .wav files instead of arrays
    :param file_name: path
    :param meth: 'hrtf' or 'room'
    :param kwargs:
    :return:
    """
    stim, stim_sr = utils.wav_to_array(file_name, rescale=None)
    return augment_from_array(stim, stim_sr, meth, **kwargs)


# Name: convolve apply signal hrtf plural; applyHRTFs? but probably should be restructured anyways
def augment_from_array(signal, sample_rate, method='hrtf', max_scaling=0.1, hrtfs=DEFAULT_HRTF_INFO, **kwargs):
    """
    - Checks for signal format
    - Normalizes
    - Calls function that convolves the signal with the HRTFs

    Given a single channel audio signal, uses an HRTF to generate binaural sounds at different locations.
    Assumes a single channel, otherwise takes the first channel.
    TODO: Implement room simulation
    :param signal: np.array, single channel audio signal
    :param sample_rate: Hz
    :param method: 'hrtf' or 'room'
    :param kwargs:
    :return: Dictionary of some kind, containing binaural sounds and labels
    """
    pass


    # hrtfs was hrtf_sets, but it's just passed through, so it shouldn't make any difference

    # Check and correct format of signal
    if len(signal.shape) == 2:  # length of shape is the number of dimensions, if 2, then it's a stereo sound
        if min(signal.shape) > 1:  # If the smaller dimension has value of >1 it means there's more than one channel
            warnings.warn('signal contains multiple channels. only take the first channel')
            if np.argmin(signal.shape) == 0:  # Get the correct dimension and pick its first channel
                signal = signal[0]
            else:
                signal = signal[:, 0]
        else:  # If the smaller dimension is 1, it's a single channel, but packed in the shape of (1, n_samples), so ravel it to make it a 1D array
            signal = signal.ravel()
    if len(signal.shape) > 2:  # More than 2 dimensions, doesn't make sense
        raise ValueError('signal has more than 2 dimensions')
    if len(signal) / sample_rate > 5:  # Raise warning if the sound is longer than 5s
        warnings.warn(f'Signal is more than 5s long ({len(signal) / sample_rate}s). May consume a lot of memory.')

    # first normalize the signal
    # TODO: does it make sense to do the normalization here?
    signal = max_scaling * utils.rescale_sound(signal, 'normalize')

    # generate localized sound
    if method == 'hrtf':
        return simulate_from_hrtf(signal, sample_rate, hrtfs, **kwargs)
    elif method == 'room':
        raise NotImplementedError('Room simulation not implemented yet')
    else:
        raise ValueError(f'method: {method} not known')


def simulate_from_hrtf(sig, sig_sample_rate, hrtf_tuple, target_sample_rate=48000, **kwargs):
    """
    - Checks and adjusts sample rate of signal and HRTFs
    - For each position in hrtf_tuple, applies the corresponding HRTF to the signal
    - Adds labels and saves the data in a dictionary which itself is saved in a list that is returned

    generate binaural sounds from different HRTFs
    :param sig: np.array, single channel audio signal
    :param sig_sample_rate: sample rate of the signal, Hz
    :param hrtf_tuple: output from pick_hrtf_by_loc, tuple: (hrtf_obj, hrtf_indices, positions)
    :param target_sample_rate: target sample rate, Hz
    :return: a list of dictionaries containing a binaural signal and its corresponding dictionary of labels
    """
    hrtf_obj, hrtf_indices, positions = hrtf_tuple
    # What is hrtf_obj? Seems to be multiple HRTFs... look at slab docs; or just because it's 2 for 2 ears? yes! but technically could be more than 2.

    hrtf_filters: slab.HRTF = deepcopy(hrtf_obj)  # Deepcopy as to not touch the original HRTF object

    # Resample if needed
    if hrtf_filters.samplerate != target_sample_rate:
        for idx, hrtf_info in enumerate(hrtf_filters):
            hrtf_filters[idx] = hrtf_info.resample(target_sample_rate)
    if sig_sample_rate != target_sample_rate:
        sig = resample(sig, target_sample_rate, sig_sample_rate, As=75, N=64001)

    # TODO: Rather use named tuples?
    sig_dicts = []
    slab_sig = slab.Sound(sig, samplerate=target_sample_rate)  # Convert to slab's Sound data type
    for hrtf_idx, pos in zip(hrtf_indices, positions):
        bi_sig = hrtf_filters.apply(hrtf_idx, slab_sig)
        # label for current stim
        lb_dict = {'azim': int(pos[0]), 'elev': int(pos[1]), 'dist': float(pos[2]), 'sampling_rate': target_sample_rate,
                   'hrtf_idx': int(hrtf_idx), 'cnn_idx': loc_to_CNNpos(pos[0], pos[1])}
        # add additional labels if needed
        if 'extra_lbs' in kwargs:
            lb_dict.update(kwargs['extra_lbs'])
        sig_dicts.append({'sig': bi_sig.data, 'label': lb_dict})

    return sig_dicts


def render_stims(orig_stim, pos_azim, pos_elev, hrtf_obj=None, n_reps=1, n_sample=None, **kwargs):
    """
    - Picks closest HRTFs for the given positions
    - Calls augment_from_array to render the stimuli
        - if one sound, applies the HRTFs for the given positions
        - if multiple sounds, randomly selects a subset and separately applies the HRTFs for the given positions
    - Adds the HRTF-applied stimuli to a list (repeating each n_reps times) along with labels and returns the list
    -> List of dictionaries, each containing a binaural signal and its corresponding label dictionary


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

    # Apply HRTFs
    if isinstance(orig_stim, slab.Sound):
        # If the original stimulus is a single sound
        stims_rendered.extend(
            augment_from_array(orig_stim.data, sample_rate=orig_stim.samplerate, hrtfs=hrtf_sets, **kwargs))
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
            sig = slab.Binaural(stim_dict['sig'],
                                samplerate=sig_rate)  # Why cast to Binaural if later we just use .data?
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
