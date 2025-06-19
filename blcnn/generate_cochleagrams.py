import datetime
import glob
import itertools
import json
import logging
import os
import pprint
import random
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Generator, Tuple
from time import strftime

import coloredlogs
import numpy as np
import scipy as sp
import slab
from slab import Filter
from tqdm import tqdm
import tensorflow as tf

from util import get_unique_folder_name, load_config, CochleagramConfig, Config, SourcePositionsConfig, loc_to_CNNpos
from legacy.CNN_preproc import cochleagram_wrapper
from generate_brirs import TrainingCoordinates, run_brir_sim, RoomConfig, calculate_listener_positions, \
    CartesianCoordinates, generate_source_positions, SphericalCoordinates
from legacy.stim_util import zero_padding, normalize_binaural_stim

# Needed bc there's a bug in slab.Signal's add method that doesn't preserve the samplerate
slab.Signal.set_default_samplerate(48000)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    generate_and_persist_cochleagrams_for_multiple_HRTFs()


def generate_and_persist_cochleagrams_for_multiple_HRTFs():
    config = load_config('blcnn/config.yml')
    logger.info(f'Loaded config: {pprint.pformat(config)}')

    use_bkgd = config.generate_cochleagrams.use_bkgd
    if use_bkgd:
        logger.error('Background noise is not yet implemented. Exiting...')
        sys.exit(1)

    # Check if the HRTF files specified in the yaml exist
    for hrtf_path in config.generate_cochleagrams.hrtf_labels:
        if not Path(f'data/brirs/{hrtf_path}').exists():
            logger.error(f'BRIRs for the HRTF {hrtf_path} specified in config.yml do not exist. Exiting...')
            sys.exit(1)

    inputs = [c for c in
              itertools.product(config.generate_cochleagrams.stim_paths, config.generate_cochleagrams.hrtf_labels)]
    logger.info(f'Found the following combinations of inputs for cochleagram generation:\n{inputs}')
    for stim_path, hrtf_label in inputs:
        generate_cochleagrams(config, Path(stim_path), hrtf_label)


def generate_cochleagrams(config: Config, stim_path: Path, hrtf_label: str):
    cochleagram_config = config.generate_cochleagrams

    start_time = time.time()
    timestamp = strftime("%Y-%m-%d_%H-%M-%S")

    if config.generate_cochleagrams.anechoic:
        dest = get_unique_folder_name(f'data/cochleagrams/{stim_path.stem}_{hrtf_label}_anechoic/')
    else:
        dest = get_unique_folder_name(f'data/cochleagrams/{stim_path.stem}_{hrtf_label}/')
    Path(dest).mkdir(parents=True, exist_ok=False)

    # Resample background sounds to 48kHz
    # for file in Path('resources/McDermott_Simoncelli_2011_168_Sound_Textures_48kHz').glob('*.wav'):
    #     slab.Sound(file).resample(48000).write(file)
    # -> Assuming now that all textures are 48kHz

    stim_paths = list(stim_path.glob('*.wav'))

    path_to_backgrounds = Path(config.generate_cochleagrams.bkgd_path)
    bkgd_paths = list(path_to_backgrounds.glob('*.wav'))

    path_to_brirs = Path(f'data/brirs/{hrtf_label}')

    options = tf.io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.GZIP)
    writer = tf.io.TFRecordWriter((dest / 'cochleagrams.tfrecord').as_posix(), options=options)

    try:
        ##### Parallel #####
        # nr_workers = multiprocessing.cpu_count()
        # div, mod = divmod(len(stim_paths), nr_workers)
        # chunksize = div + 1 if mod else div
        #
        # logging.info(f'Using {nr_workers} workers')
        # logging.info(f'Chunksizes: {chunksize}')
        #
        # training_samples = []
        # with Pool(nr_workers) as pool:
        #     for samples_from_one_sound in tqdm(pool.imap_unordered(generate_training_samples_from_stim_path, stim_paths, chunksize=chunksize),
        #                                        desc='Raw sounds transformed', total=len(stim_paths), position=0):
        #         training_samples.extend(samples_from_one_sound)
        #         bar.update(len(samples_from_one_sound))

        ##### Sequential #####
        # global inner_bar
        # inner_bar = tqdm(desc='Generated training samples', position=1, unit='samples', leave=False)
        for single_stim_path in tqdm(stim_paths, desc='Processed stim paths', position=0, unit='paths',
                                     total=len(stim_paths)):
            if config.generate_cochleagrams.anechoic:
                for training_sample, training_coords in generate_training_sample_from_stim_path_anechoic(config,
                                                                                                         single_stim_path,
                                                                                                         hrtf_label):
                    write_tfrecord(training_sample, training_coords, single_stim_path.name, writer)
            else:
                for training_sample, training_coords in generate_training_samples_from_stim_path(config,
                                                                                                 single_stim_path,
                                                                                                 path_to_brirs=path_to_brirs):
                    write_tfrecord(training_sample, training_coords, single_stim_path.name, writer)
                    # inner_bar.update(1)
        # inner_bar.close()
    except Exception as e:
        logger.error(f'An error occurred during BRIR generation: {e}\n'
                     f'{traceback.print_exc()}')
    finally:
        writer.close()

        elapsed_time = str(datetime.timedelta(seconds=time.time() - start_time))
        summary = summarize_cochleagram_generation_info(cochleagram_config, hrtf_label, timestamp, elapsed_time, dest)
        logger.info(summary)
        with open(dest / f'_summary_{timestamp}.txt', 'w') as f:
            f.write(summary)


def summarize_cochleagram_generation_info(cochleagram_config: CochleagramConfig,
                                          hrtf_label: str,
                                          timestamp: str,
                                          elapsed_time: str,
                                          dest: Path) -> str:
    # Load BRIR summary
    path_to_brirs = Path(f'data/brirs/{hrtf_label}')
    with open(glob.glob((path_to_brirs / '_summary_*.txt').as_posix())[0], 'r') as f:
        brir_summary = f.read()

    summary = f'##### COCHLEAGRAM GENERATION INFO #####\n' \
              f'HRTF label: {hrtf_label}\n' \
              f'Timestamp: {timestamp}\n\n' \
              f'Total elapsed time: {elapsed_time}\n' \
              f'Number of BRIRs found: {len(list(path_to_brirs.glob("brir_*")))}\n' \
              f'Number of Stimuli found (only if a single folder is specified): {len(list(glob.glob(f"{cochleagram_config.stim_paths}/*.wav")))}\n' \
              f'Number of Backgrounds found: {len(list(glob.glob(f"{cochleagram_config.bkgd_path}/*.wav")))}\n' \
              f'Config:\n{pprint.pformat(cochleagram_config)}\n\n' \
              f'Based on the following BRIR generation:\n' \
              f'{brir_summary}\n\n' \
              f'################################\n' \
              f'Cochleagrams saved to: {dest}\n' \
              f'################################\n'
    return summary


def generate_training_samples_from_stim_path(config: Config,
                                             stim_path: Path,
                                             brir_dict: Dict[TrainingCoordinates, slab.Filter] = None,
                                             path_to_brirs: Path = Path('data', 'brirs_2024-09-13_14-13-42'),
                                             no_bkgd=True
                                             ) -> List[Tuple[np.ndarray, TrainingCoordinates]]:
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
    # path_to_brirs = Path('data', 'brirs_2024-09-13_14-13-42')
    # global inner_bar

    raw_stim = slab.Sound(stim_path).resample(48000)
    # raw_stim.play()
    # augmented_sounds = augment_raw_sound(raw_stim)  # Produces multiple sounds
    # for s in augmented_sounds:
    #     s.play()
    augmented_sounds = [raw_stim]

    source_positions = generate_source_positions(config.generate_cochleagrams.source_positions)
    stim_generator = generate_spatialized_sound(augmented_sounds, config.generate_brirs.room_configs, source_positions,
                                                brir_dict=brir_dict, path_to_brirs=path_to_brirs)

    training_samples = []
    # worker_nr = int(multiprocessing.current_process().name.split('-')[-1])
    # for spatialized_sound, training_coordinates in tqdm(stim_generator, desc= f'Process {worker_nr}',position=worker_nr, leave=False):
    for spatialized_sound, training_coordinates in tqdm(stim_generator, desc='Generated training samples', position=1,
                                                        unit='samples', leave=False):
        normalized_sound = spatialized_sound * (0.1 / np.max(np.abs(spatialized_sound.data)))
        if no_bkgd:
            training_samples.append((transform_stim_to_cochleagram(normalized_sound), training_coordinates))
        else:
            background = create_background(training_coordinates.room_id,
                                           training_coordinates.listener_position,
                                           source_positions,
                                           brir_dict=brir_dict,
                                           path_to_brirs=path_to_brirs)
            snr_factor = (10 ** (-random.uniform(5, 30) / 20))
            combined_sound = normalized_sound + background * snr_factor
            training_samples.append((transform_stim_to_cochleagram(combined_sound), training_coordinates))
        # inner_bar.update(1)
    return training_samples


def generate_training_sample_from_stim_path_anechoic(config: Config, stim_path: Path, hrtf_label: str):
    src_positions = [c for c in itertools.product(config.generate_cochleagrams.source_positions.azimuths,
                                                  config.generate_cochleagrams.source_positions.elevations)]

    # Go through sounds in data/raw/uso_500ms_raw and apply the HRTFs
    # for sound_path in tqdm(Path('data/raw/uso_500ms_raw').glob('*.wav'), desc='Sounds', position=0):
    sound = slab.Sound(stim_path).resample(48000)
    padded_sound = zero_padding(sound, goal_duration=2, type="frontback")
    training_samples = []
    for (azim, elev) in tqdm(src_positions, desc='HRTFs', position=1, leave=False):
        # 20% chance to use this HRTF
        if random.random() <= 1.0:
            hrtf_sound = interpolate_HRTF(hrtf_label, azim, elev).apply(padded_sound)
            cochleagram = transform_stim_to_cochleagram(hrtf_sound)
            training_samples.append(
                (cochleagram, TrainingCoordinates(0, CartesianCoordinates(0, 0), SphericalCoordinates(azim, elev))))
        print(interpolate_HRTF.cache_info(), hrtf_label, azim, elev, end='\r')
    return training_samples


# Cache useful bc the same HRTF set is used multiple times for different sounds
# -> Takes almost same time as without cache...
@lru_cache(maxsize=550)
def interpolate_HRTF(hrtf_label: str, azim: int, elev: int) -> slab.Filter:
    """
    Interpolates the HRTF at the given azimuth and elevation.
    Cached to avoid multiple interpolations with the same parameters.
    Args:
        hrtf_label: Label of the HRTF
        azim: Azimuth
        elev: Elevation
    """
    if hrtf_label == 'slab_kemar':
        loaded_hrtf = slab.HRTF.kemar()
    else:
        loaded_hrtf = slab.HRTF(f'data/hrtfs/{hrtf_label}.sofa', verbose=True)
    return loaded_hrtf.interpolate(azim, elev).resample(48000)


def write_tfrecord(cochleagram, training_coords, stim_file_name: str, writer):
    """
    Write a training sample to a tfrecord file
    Args:
        cochleagram:
        training_coords:
        writer:

    Returns:

    """
    # TODO: Doesn't shuffle data
    target = loc_to_CNNpos(training_coords.source_position.azim, training_coords.source_position.elev)

    data = {
        'train/image': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(cochleagram.tobytes())])),
        # 'train/image_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[cochleagram.shape[0]])),
        # 'train/image_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[cochleagram.shape[1]])),
        # 'train/azim': tf.train.Feature(int64_list=tf.train.Int64List(value=[training_coords.source_position.azim])),
        # 'train/elev': tf.train.Feature(int64_list=tf.train.Int64List(value=[training_coords.source_position.elev])),
        'train/target': tf.train.Feature(int64_list=tf.train.Int64List(value=[target])),
        # 'train/name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[stim_file_name.encode('utf-8')]))
    }

    # write the single record into tfrecord file
    example = tf.train.Example(features=tf.train.Features(feature=data))
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())


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


def generate_spatialized_sound(sounds: List[slab.Sound],
                               room_configs: List[RoomConfig],
                               source_positions: List[SphericalCoordinates],
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
        for training_coordinates in generate_training_locations(room_configs, source_positions):
            spatialized_sound = apply_brir(padded_sound, training_coordinates, brir_dict=brir_dict,
                                           path_to_brirs=path_to_brirs)
            # PBAR.update(1)
            if spatialized_sound is not None:
                yield spatialized_sound, training_coordinates


def create_background(room_id: int,
                      listener_position: CartesianCoordinates,
                      source_positions: List[SphericalCoordinates],
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
        source_positions:
        room_id:
        brir_dict:

    Returns:

    """
    rand_texture_path = random.choice(
        list(Path('data/raw/McDermott_Simoncelli_2011_168_Sound_Textures_48kHz').glob('*.wav')))
    background_textures = []

    for _ in range(random.randint(3, 8)):
        rand_start = random.uniform(0, 3)
        texture = slab.Sound(rand_texture_path).trim(rand_start, rand_start + 2.0)
        # For spatialization we need: room_id, listener location; we pick source location randomly
        # TODO: Get source positions from somewhere else.
        #  - not from config; config file may have changed
        #  - not from summary file of BRIR generation; ideally the summary shouldn't be used as a data source
        #    -> Might be most straightforward and simple though
        #  - maybe save the source positions to a file associated with the run
        random_location = random.choice(source_positions)
        spatialized_texture = apply_brir(texture,
                                         TrainingCoordinates(room_id,
                                                             listener_position,
                                                             random_location),
                                         brir_dict=brir_dict,
                                         path_to_brirs=path_to_brirs)
        if spatialized_texture is not None:
            background_textures.append(spatialized_texture)
    # Need to supply starting sound for sum on which to add the textures
    summed_textures = sum(background_textures, start=slab.Sound(np.zeros_like(background_textures[0].data)))
    normalized_background = summed_textures * (0.99 / np.max(
        np.abs(summed_textures.data)))  # no attenuation here; 0.99 to avoid technical errors when persisting
    return normalized_background


def generate_training_locations(room_configs: List[RoomConfig], source_positions: List[SphericalCoordinates]) -> \
Generator[TrainingCoordinates, None, None]:
    #  Dict[int, RoomConfig]
    nr_listener_positions_smallest_room = min(
        [len(calculate_listener_positions(room.width, room.length)) for room in room_configs])

    # for augmented_sound in tqdm(range(2492)):  # ca. 31s for 2492 locations (for one sound)
    for room in room_configs:
        listener_positions_current_room = calculate_listener_positions(room.width, room.length)
        for listener_position in listener_positions_current_room:
            for source_position in source_positions:
                # if random.random() < (0.025 * nr_listener_positions_smallest_room) / len(listener_positions):
                # if random.random() < (0.2 * nr_listener_positions_smallest_room) / len(listener_positions_current_room):
                if random.random() < (0.05 * nr_listener_positions_smallest_room) / len(listener_positions_current_room):
                    # Normalization works: Rooms are equally represented
                    # Nr. of total locations is too big though 628k vs 545k in paper
                    yield TrainingCoordinates(room.id, listener_position, source_position)


def apply_brir(sound: slab.Sound,
               training_coordinates: TrainingCoordinates,
               brir_dict: Dict[TrainingCoordinates, slab.Filter] = None,
               path_to_brirs=None) -> slab.Signal | None:
    """
    Applies the BRIR to the given sound at the given training coordinates.
    If a BRIR dictionary is given, the BRIR is applied from the dictionary, otherwise it is calculated on the fly.
    Args:
        brir_dict: Dictionary containing the BRIRs
        training_coordinates: TrainingCoordinates object containing the room ID, listener position and source position
        sound: slab.Sound object to which the BRIR should be applied
        path_to_brirs: Path to the BRIRs

    Returns:
        slab.Sound object with the BRIR applied
    """
    # switch case
    if path_to_brirs:
        try:
            return Filter.load(Path(path_to_brirs, f'brir_{training_coordinates}.wav.npy')).apply(sound).trim(0.0, 2.0)
        except FileNotFoundError as e:
            logger.warning(f'An error occurred during BRIR application: {e}\n'
                           f'Probably the BRIR file for {training_coordinates} does not exist.')
            return None
    elif brir_dict:
        # TODO: Handle if brir not in dict
        return brir_dict[training_coordinates].apply(sound).trim(0.0, 2.0)
    else:
        return run_brir_sim(training_coordinates)[1].resample(48000).apply(sound).trim(0.0, 2.0)


# DOWNSAMPLE_FILTER = make_downsample_filter()


def transform_stim_to_cochleagram(stim: slab.Binaural):
    """

    Args:
        stim:

    Returns:

    """
    normalized_stim, sr = normalize_binaural_stim(stim.data, stim.samplerate)
    cochleagram = cochleagram_wrapper(normalized_stim)
    # Downsample to 8kHz
    # logger.info(type(cochleagram))
    downsampled = downsample_hardcoded(cochleagram).numpy()
    # logger.info(type(downsampled))

    # -> low subband index is low frequency band; plot cochleagrams them upside down
    return downsampled

    # print(cochleagram.shape, type(cochleagram))
    # downsampled = slab.Signal(cochleagram.T).resample(8000)
    # -> Doesn't work bc Signal() automatically transposes the cochleagram, assuming it's a <=2D signal
    # print(sp.signal.resample(cochleagram, 8000, axis=1).shape, type(sp.signal.resample(cochleagram, 8000, axis=1)))
    # downsampled = sp.signal.resample(cochleagram, 8000, axis=1)
    # -> 0.031 per call, 12.315 tottime for 100 calls
    # downsampled = resample(cochleagram, 8000, 48000, axis=1, fc='nn')
    # -> hangs up when calling upfirdn

    # return downsampled
    # Shape: (39, 48000, 2)
    # Downsample to: (39, 8000, 2)
    # TODO: Figure out dimension stuff; why is filter (1, filt_len, 2, 2)? What does stride=[1, 1, ds_ratio, 1] mean (the ones)?


def make_downsample_filt_tensor_hardcoded():
    downsample_filter_times = np.arange(-4097 / 2, int(4097 / 2))
    downsample_filter_response_orig = np.sinc(downsample_filter_times / 6) / 6
    downsample_filter_window = sp.signal.windows.kaiser(4097, 10.06)
    downsample_filter_response = downsample_filter_window * downsample_filter_response_orig
    downsample_filt_tensor = tf.constant(downsample_filter_response, tf.float32)
    downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 0)
    downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 2)
    downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 3)
    return downsample_filt_tensor


DS_KERNEL = make_downsample_filt_tensor_hardcoded()


def downsample_hardcoded(signal):
    [L_channel, R_channel] = tf.unstack(signal, axis=2)
    concat_for_downsample = tf.concat([L_channel, R_channel], axis=0)
    reshaped_for_downsample = tf.expand_dims(concat_for_downsample, axis=2)

    signal = tf.expand_dims(reshaped_for_downsample, 0)
    downsampled_signal = tf.nn.conv2d(signal, DS_KERNEL, strides=[1, 1, 6, 1], padding='SAME',
                                      name='conv2d_cochleagram_raw')
    downsampled_signal = tf.nn.relu(downsampled_signal)

    downsampled_squeezed = tf.squeeze(downsampled_signal)
    [L_channel_downsampled, R_channel_downsampled] = tf.split(downsampled_squeezed, num_or_size_splits=2, axis=0)
    downsampled_reshaped = tf.stack([L_channel_downsampled, R_channel_downsampled], axis=2)
    downsampled_signal = tf.pow(downsampled_reshaped, 0.3)

    return downsampled_signal


"""
- Probably using TF because 2d conv is faster than filtering all cochleagram channels separately
-> Ideally profile TF 2d conv vs. nnresample vs. scipy.signal.resample vs. scipy.signal.fftconvolve
-> Also look at slab.Signal.resample(), could use it on cochleagram w/ many channels
-> Should this speed comparison be in the thesis? Probably not, right?

- Only decimating to 8kHz introduces aliasing, so we apply a filter before

"""


def profile_transform_stim_to_cochleagram():
    stim = slab.Binaural(slab.Sound.pinknoise(duration=3.0, samplerate=48000))
    import cProfile, pstats
    with cProfile.Profile() as pr:
        for _ in range(100):
            print(_)
            transform_stim_to_cochleagram(stim)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats(filename='profile_stats_cochleagram.prof')


def plot_cochleagram(cochleagram):
    import matplotlib.pyplot as plt
    # Shape of cochleagram: (39, 48000, 2)

    # left = cochleagram[:, 2200:2400, 0]  # Plot shorter interval so smoothing doesn't cancel out the high and low points in signal
    # right = cochleagram[:, 2200:2400, 1]
    left = cochleagram[:, :, 0]
    right = cochleagram[:, :, 1]
    fig, axs = plt.subplots(2, 1, figsize=(20, 10))
    axs[0].imshow(left, aspect='auto', cmap='PuOr', origin='lower')
    axs[1].imshow(right, aspect='auto', cmap='PuOr', origin='lower')
    plt.show()


def plot_slab_cochleagram(cochleagram):
    import matplotlib.pyplot as plt
    # Shape of cochleagram: (120000, 39)
    plt.imshow(cochleagram.T, aspect='auto', cmap='PuOr', origin='lower')
    plt.show()


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

McDermott's website:
- Datasets for stimuli and textures
- Matlab Code for texture generation!
- pychochleagram code for vanilla python, tensorflow, and pytorch

Notes:
- My BRIR gen is much faster than in the paper. Unsure why.


##### PROFILING #####
-> sequential: total time 30 source files: 4:27 min
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
198656  100.560    0.001  100.560    0.001 {built-in method scipy.fft._pocketfft.pypocketfft.r2c}
 99328   52.136    0.001   52.136    0.001 {built-in method scipy.fft._pocketfft.pypocketfft.c2r}
 42075   29.815    0.001   30.042    0.001 /Users/david/Repositories/ma/BinauralLocalizationCNN/venv_stim_gen/lib/python3.11/site-packages/soundfile.py:1346(_cdata_io)
 49664   12.149    0.000   12.190    0.000 {built-in method numpy.fromfile}
198660   10.144    0.000   10.144    0.000 {method 'read' of '_io.BufferedReader' objects}
213894    7.673    0.000    7.673    0.000 {method '__deepcopy__' of 'numpy.ndarray' objects}
 49664    5.458    0.000  177.352    0.004 /Users/david/Repositories/ma/BinauralLocalizationCNN/venv_stim_gen/lib/python3.11/site-packages/slab/filter.py:138(apply)
397402    4.342    0.000    4.342    0.000 {built-in method numpy.array}

"""

if __name__ == "__main__":
    main()
