import logging
import pickle
import random
import sys

import numpy as np
from typing import List, Iterable, Generator, Dict, Tuple

from slab import Filter

from generate_brirs import CartesianCoordinates, RoomConfig, TrainingCoordinates, MCDERMOTT_SOURCE_POSITIONS, \
    MCDERMOTT_ROOM_CONFIGS, run_brir_sim, calculate_listener_positions
from stim_util import loc_to_CNNpos, zero_padding
from nnresample import resample
import slab
# import pyroomacoustics as pra
from copy import deepcopy
from stim_manipulation import change_itd
from pycochleagram import utils
import itertools
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path

# PBAR = tqdm()

# use Kemar HRTF
KEMAR_HRTF = slab.HRTF.kemar()
KEMAR_SAMPLERATE = KEMAR_HRTF.samplerate
# KEMAR_48KHZ: slab.HRTF = slab.HRTF.kemar().resample(48000)
# source locations are in hrtf.sources, centered on listener


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
# DEFAULT_HRTF_INFO = pick_hrtf_by_loc(*DEFAULT_POSITIONS, )
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


#     # hrtfs was hrtf_sets, but it's just passed through, so it shouldn't make any difference
#
#     # Check and correct format of signal
#     if len(signal.shape) == 2:  # length of shape is the number of dimensions, if 2, then it's a stereo sound
#         if min(signal.shape) > 1:  # If the smaller dimension has value of >1 it means there's more than one channel
#             warnings.warn('signal contains multiple channels. only take the first channel')
#             if np.argmin(signal.shape) == 0:  # Get the correct dimension and pick its first channel
#                 signal = signal[0]
#             else:
#                 signal = signal[:, 0]
#         else:  # If the smaller dimension is 1, it's a single channel, but packed in the shape of (1, n_samples), so ravel it to make it a 1D array
#             signal = signal.ravel()
#     if len(signal.shape) > 2:  # More than 2 dimensions, doesn't make sense
#         raise ValueError('signal has more than 2 dimensions')
#     if len(signal) / sample_rate > 5:  # Raise warning if the sound is longer than 5s
#         warnings.warn(f'Signal is more than 5s long ({len(signal) / sample_rate}s). May consume a lot of memory.')
#
#     # first normalize the signal
#     # TODO: does it make sense to do the normalization here?
#     signal = max_scaling * utils.rescale_sound(signal, 'normalize')
#
#     # generate localized sound
#     if method == 'hrtf':
#         return simulate_from_hrtf(signal, sample_rate, hrtfs, **kwargs)
#     elif method == 'room':
#         raise NotImplementedError('Room simulation not implemented yet')
#     else:
#         raise ValueError(f'method: {method} not known')


def simulate_from_room():
    pass


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


def augment_raw_sound(sound: slab.Sound, lowest_center_freq=100, nr_octaves=8) -> List[slab.Sound]:
    """
    Given a generator of slab.Sound objects, increases the number of sounds by applying bandpass filters.
    TODO: Check if it's a 2nd order Butterworth filter

    (- "All sounds were sampled at 44.1 kHz.")
    (- 455 sounds, 385 for training, 70 for validation and testing)
    - Augmented by applying bandpass filters: two-octave wide, second-order Butterworth filters w/ center
    frequencies spaced in one-octave steps from 100Hz. (Up to?)
    -> Yielded 2492 =~ 455 * 5.477 sounds; doesn't make sense, again...

    Args:
        sound: slab.Sound object to augment

    Returns:
        List of slab.Sound objects
    """
    augmented_sounds = []
    for octave_index in range(nr_octaves):
        center_freq = lowest_center_freq * 2 ** octave_index
        low_freq = center_freq / 2
        high_freq = min(center_freq * 2, (sound.samplerate / 2) - 1)  # Bit hacky with the -1
        # print(f'Center Frequency: {center_freq} Hz, Low Frequency: {low_freq} Hz, High Frequency: {high_freq} Hz')
        augmented_sounds.append(sound.filter(frequency=(low_freq, high_freq), kind='bp'))
    return augmented_sounds


def apply_brir(sound: slab.Sound,
               training_coordinates: TrainingCoordinates,
               brir_dict: Dict[TrainingCoordinates, slab.Filter] = None,
               path_to_brirs=None) -> slab.Sound:
    """
    Applies the BRIR to the given sound at the given training coordinates.
    If a BRIR dictionary is given, the BRIR is applied from the dictionary, otherwise it is calculated on the fly.
    Args:
        brir_dict: Dictionary containing the BRIRs
        training_coordinates: TrainingCoordinates object containing the room ID, listener position and source position
        sound: slab.Sound object to which the BRIR should be applied

    Returns:
        slab.Sound object with the BRIR applied
    """
    # switch case
    if path_to_brirs:
        return Filter.load(Path(path_to_brirs, f'brir_{training_coordinates}.wav.npy')).apply(sound).trim(0.0, 2.0)
    elif brir_dict:
        # TODO: Handle if brir not in dict
        return brir_dict[training_coordinates].apply(sound).trim(0.0, 2.0)
    else:
        return run_brir_sim(training_coordinates)[1].resample(48000).apply(sound).trim(0.0, 2.0)


def generate_training_locations(room_configs: Dict[int, RoomConfig]) -> Generator[TrainingCoordinates, None, None]:
    nr_listener_positions_smallest_room = min(
        [len(calculate_listener_positions(room_config.room_size)) for room_id, room_config in room_configs.items()])

    # for augmented_sound in tqdm(range(2492)):  # ca. 31s for 2492 locations (for one sound)
    for room_id, room_config in room_configs.items():
        listener_positions = calculate_listener_positions(room_config.room_size)
        for listener_position in listener_positions:
            for source_position in MCDERMOTT_SOURCE_POSITIONS:
                if random.random() < (0.025 * nr_listener_positions_smallest_room) / len(listener_positions):
                    # Normalization works: Rooms are equally represented
                    # Nr. of total locations is too big though 628k vs 545k in paper
                    yield TrainingCoordinates(room_id, listener_position, source_position)


def create_background(room_id: int,
                      listener_position: CartesianCoordinates,
                      brir_dict: Dict[TrainingCoordinates, slab.Filter] = None,
                      path_to_brirs: Path = None) -> slab.Sound:
    """
    Creates a background texture by summing up 3-8 randomly chosen textures.
    TODO: Implement texture synthesis, either here online, or offline in Matlab
    -> For now: Load one 5s texture, cut up randomly to yield nr_samples of 2s textures

    - Pick random texture class
    - Yield 3-8 random texture samples from that class
    - Pick 3-8 of random coordinates (absolute on grid or relative around listener? unclear, go for relative first)
    - Spatialize them
    - Add them together and yield

    Args:
        path_to_brirs:
        listener_position:
        room_id:
        brir_dict:

    Returns:

    """
    rand_texture_path = random.choice(
        list(Path('resources', 'McDermott_Simoncelli_2011_168_Sound_Textures').glob('*.wav')))
    background_textures = []

    for _ in range(random.randint(3, 8)):
        rand_start = random.uniform(0, 3)
        texture = slab.Sound(rand_texture_path).trim(rand_start, rand_start + 2.0)
        # For spatialization we need: room_id, listener location; we pick source location randomly
        random_location = random.choice(MCDERMOTT_SOURCE_POSITIONS)
        spatialized_texture = apply_brir(texture,
                                         TrainingCoordinates(room_id,
                                                             listener_position,
                                                             random_location),
                                         brir_dict=brir_dict,
                                         path_to_brirs=path_to_brirs)
        background_textures.append(spatialized_texture)
    # Need to supply starting sound for sum on which to add the textures
    summed_textures = sum(background_textures, start=slab.Sound(np.zeros_like(background_textures[0].data)))
    normalized_background = summed_textures * (0.1 / np.max(np.abs(summed_textures.data)))
    return normalized_background


def generate_spatialized_sound(sounds: List[slab.Sound],
                               brir_dict: Dict[TrainingCoordinates, slab.Filter] = None,
                               path_to_brirs: Path = None) -> Generator[
    Tuple[slab.Sound, TrainingCoordinates], None, None]:
    """
    - sound generator
        - Go through all TrainingCoordinates, for all randomly picked ones:
            - pick random sound sample and spatialize it
            - yield sample + label
    Returns:

    """
    for sound in sounds:
        padded_sound = zero_padding(sound, goal_duration=2, type="frontback")
        # Render sound at different positions
        for training_coordinates in generate_training_locations(MCDERMOTT_ROOM_CONFIGS):
            spatialized_sound = apply_brir(padded_sound, training_coordinates, brir_dict=brir_dict,
                                           path_to_brirs=path_to_brirs)
            # PBAR.update(1)
            yield spatialized_sound, training_coordinates


def generate_training_samples_from_stim_path(stim_path: Path,
                                             brir_dict: Dict[TrainingCoordinates, slab.Filter] = None,
                                             path_to_brirs: Path = None
                                             ) -> List[Tuple[slab.Sound, TrainingCoordinates]]:
    #     -> Generator[
    # Tuple[slab.Sound, TrainingCoordinates], None, None]:
    """
    - generator to combine sound sources and background noise -> yield sample
        - for all spatialized samples
            - get a noise scene and add them together with a SNR uniformly sampled from 5-30dB
            - normalize to 0.1 r.m.s.
            - yield combined sample + label
    Returns:

    """
    path_to_brirs = Path('data', 'brirs_2024-09-13_14-13-42')

    raw_stim = slab.Sound(stim_path)
    # raw_stim.play()
    # augmented_sounds = augment_raw_sound(raw_stim)  # Produces multiple sounds
    # for s in augmented_sounds:
    #     s.play()
    augmented_sounds = [raw_stim]

    stim_generator = generate_spatialized_sound(augmented_sounds, brir_dict=brir_dict, path_to_brirs=path_to_brirs)

    training_samples = []
    # worker_nr = int(multiprocessing.current_process().name.split('-')[-1])
    # for spatialized_sound, training_coordinates in tqdm(stim_generator, desc= f'Process {worker_nr}',position=worker_nr, leave=False):
    for spatialized_sound, training_coordinates in stim_generator:
        normalized_sound = spatialized_sound * (0.1 / np.max(np.abs(spatialized_sound.data)))
        background = create_background(training_coordinates.room_id, training_coordinates.listener_position,
                                       brir_dict=brir_dict, path_to_brirs=path_to_brirs)
        snr_factor = (10 ** (-random.uniform(5, 30) / 20))  # TODO: No idea if this is right...
        combined_sound = normalized_sound + background * 0.1
        # yield combined_sound, training_coordinates
        training_samples.append((combined_sound, training_coordinates))
    return training_samples


def main():
    logging.basicConfig(level=logging.INFO)
    # Resample background sounds to 48kHz
    # for file in Path('resources', 'McDermott_Simoncelli_2011_168_Sound_Textures').glob('*.wav'):
    #     slab.Sound(file).resample(48000).write(file)
    # -> Assuming now that all textures are 48kHz

    # generate_and_persist_BRIRs(MCDERMOTT_ROOM_CONFIGS)
    # sys.exit(0)

    # brir_dict = pickle.load(open('brir_dict_2024-09-11_13-51-58.pkl', 'rb'))
    stim_paths = list(Path('uso_500ms_raw').glob('*.wav'))
    path_to_brirs = Path('data', 'brirs_2024-09-13_14-13-42')
    print(stim_paths)
    # One process
    # i = 0
    # for sample, training_coordinates in tqdm(generate_training_samples_from_stim_path(stim_paths[10], path_to_brirs=path_to_brirs)):
    #     if i % 20 == 0:
    #         sample.play(blocking=True)
    #     i += 1

    # NEXT STEPS
    # TODO: apply cochleagram
    # TODO: save to tfrecord
    # TODO: Run CNN evaluation on multiple datasets with different HRTFs, and see if elevation collapses

    # TODO: compare with McDermott's data generation pipeline

    bar = tqdm(desc='Generated training samples', position=1, unit='samples')
    # Parallel
    training_samples = []
    with Pool() as pool:
        for samples_from_one_sound in tqdm(pool.imap_unordered(generate_training_samples_from_stim_path, stim_paths), desc='Raw sounds transformed', total=len(stim_paths), position=0):

            training_samples.extend(samples_from_one_sound)
            bar.update(len(samples_from_one_sound))

    bar.close()


    # print_data_generation_info({t: MCDERMOTT_ROOM_CONFIGS[t]})
    # PBAR.close()
    sys.exit(0)

    # brir_dict = generate_BRIRs({t: MCDERMOTT_ROOM_CONFIGS[t]})
    brir_dict = generate_BRIRs(MCDERMOTT_ROOM_CONFIGS)
    brir_dict = pickle.load(open('brir_dict_2024-09-11_13-51-58.pkl', 'rb'))
    # print(f'Dict size: {get_deep_size(brir_dict)} bytes')

    sys.exit(0)

    # Assuming that for each augmented sound the positions and background noises are chosen independently

    """
    GOAL: A generator that spits out one labeled training sample at a time in the form of a tfrecord Example (I think)
    First: Not a generator, but a small dataset that can be saved to a tfrecord; take care of continuous data generation later.

    TODO:
    - function to generate background noise samples -> See if easier to run entirely in Matlab
        - Calls Matlab code to generate textures
        - Params for one sample -> texture transformation: samplerate, nr_textures (1000), duration (5s) 
        - Input: List of sound samples (what format?)
        - Cut down to 2s
        - Note: Picked 50 sound samples that made textures that were good (see paper)
        - Save them (ca. 8GB)
    -> Later, for now just use textures from files
    
    - BRIR on bandpassed stimuli sounds weird (big blob in bass), figure out what's going on



    - function that transforms sample into cochleagram
    - cut cochleagrams, more info in paper, maybe already implemented
    - note info and format of labels: image, image_width, image_height, elev, azim
    - First: save to tfrecord and see how much space it takes (later: maybe use online augmentation)
    - Test if data can be used to train the CNN


    Data! https://mcdermottlab.mit.edu/downloads.html
    165 2-sec sounds: https://mcdermottlab.mit.edu/svnh/Natural-Sound/Stimuli.html
    168 7-sec textures, although 20kHz
    Matlab Code for texture generation!
    pychochleagram code for vanilla python, tensorflow, and pytorch



    Notes:
    - My BRIR gen is much faster than in the paper. Unsure why.
    """

    # from stim_util import zero_padding
    # from show_subbands import show_subbands
    #
    # samplerate = 44100  # initial samplerate for CNN
    # sound = slab.Sound.pinknoise(0.5, samplerate=samplerate)
    # sound = zero_padding(sound, goal_duration=2.1, type="frontback")
    #
    # pos_azim = [-90, 0, 90]
    # pos_elev = [0]
    # stims_rendered = []  # Store the rendered stimuli
    #
    # hrtf_sets = pick_hrtf_by_loc(pos_azim=pos_azim, pos_elev=pos_elev, hrtf_obj=KEMAR_HRTF)
    # stims_rendered.extend(augment_from_array(sound.data, sample_rate=sound.samplerate, hrtfs=hrtf_sets))
    #
    # for i, stm in enumerate(stims_rendered):
    #     label = stm["label"]
    #     print(f"Stim {i}: {label}")
    #     slab.Binaural(stm["sig"], samplerate=samplerate).play()
    #     show_subbands(slab.Binaural(stm["sig"], samplerate=samplerate))
    #     plt.show(block=True)


if __name__ == "__main__":
    main()
