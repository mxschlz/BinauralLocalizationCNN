import logging
import pickle
import random
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import List, Dict, Generator, Tuple

import numpy as np
import slab
from slab import Filter
from tqdm import tqdm

from generate_brirs import TrainingCoordinates, run_brir_sim, RoomConfig, calculate_listener_positions, \
    MCDERMOTT_SOURCE_POSITIONS, CartesianCoordinates, MCDERMOTT_ROOM_CONFIGS
from stim_util import zero_padding


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
        for samples_from_one_sound in tqdm(pool.imap_unordered(generate_training_samples_from_stim_path, stim_paths),
                                           desc='Raw sounds transformed', total=len(stim_paths), position=0):
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


if __name__ == "__main__":
    main()
