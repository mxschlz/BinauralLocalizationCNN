import datetime
import itertools
import logging
import pickle
import sys
import time
import traceback
from math import floor
from multiprocessing import Pool
from pathlib import Path
from time import strftime
from typing import NamedTuple, List, Dict, Generator, Tuple
import pprint

import coloredlogs
import slab
from tqdm import tqdm

from persistent_cache import persistent_cache
from util import get_unique_folder_name, load_config, BRIRConfig, RoomConfig, SourcePositionsConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

"""
Listener height and source distance are always 1.4m, so they are not included in the NamedTuples.
"""


class CartesianCoordinates(NamedTuple):
    x: float
    y: float


class SphericalCoordinates(NamedTuple):
    azim: float
    elev: float


class TrainingCoordinates(NamedTuple):
    room_id: int
    listener_position: CartesianCoordinates
    source_position: SphericalCoordinates

    def __str__(self):
        return f'{self.room_id}r_{self.listener_position.x}x_{self.listener_position.y}y_{self.source_position.elev}e_{self.source_position.azim}a'


"""
Francl used this KEMAR HRTF: https://sound.media.mit.edu/resources/KEMAR.html
- it's @44.1kHz, same as default slab KEMAR
- From the documentation:
    Elevation and azimuth angles indicate the location of the source
    relative to the KEMAR, such that elevation 0 azimuth 0 is directly in
    front of the KEMAR, elevation 90 is directly above the KEMAR,
    elevation 0 azimuth 90 is directly to the right of the KEMAR, etc.
    -> contrast to counter-clockwise azimuth from 0 to 355 as used in slab, and in pictures in Francl's paper
    
-> For KEMAR I'll use the default one bc direction is already correct
-> I'll use hrtf_b_nh2.sofa, which goes from 0º-0º ccw, so it should produce the correct results with slab's coordinates
(hrtf_nh2.sofa also seems to work)
"""


# Maybe setting the default samplerate makes the resampling downstairs unnecessary?
# slab.Signal.set_default_samplerate(48000)
# CUSTOM_HRTF = slab.HRTF.kemar()


def main() -> None:
    run_with_multiple_hrtfs()
    # TODO: Figure out a way to run this with one HRTF for SLURM parallelization


def run_with_multiple_hrtfs():
    brir_config = load_config('blcnn/config.yml').generate_brirs
    logger.info(f'Loaded config: {pprint.pformat(brir_config)}')

    # Check if the HRTF files specified in the yaml exist
    for hrtf_path in brir_config.hrtfs:
        if hrtf_path == 'slab_kemar':
            continue
        elif not Path(hrtf_path).exists():
            logger.error(f'HRTF file {hrtf_path} specified in config.yml does not exist. Exiting...')
            sys.exit(1)

    for hrtf_path in brir_config.hrtfs:
        generate_and_persist_BRIRs_for_single_HRTF(brir_config, hrtf_path)


def generate_and_persist_BRIRs_for_single_HRTF(brir_config: BRIRConfig, hrtf_path: str) -> None:
    """
    Given the configuration and the path to an HRTF file, generates the corresponding BRIRs and stores them in a dictionary.
    The BRIRs can be accessed using (room_id, listener_position, source_position).
    Part of reimplementation of McDermott's data generation pipeline.

    Note: I think HRIR here is really a BRIR. Assuming this for now. Technically, it's a Filter object in slab.
    Note: 64 bytes for one room.hrir() Filter object.
    Note: Dict keeps the correct filters.

    Args:
        brir_config: BRIRConfig object containing the room configurations and source positions
        hrtf_path: Path (str) to the HRTF file to use or 'slab_kemar' for the default KEMAR HRTF
    Returns:
        Dictionary with keys (room_id, listener_position, source_position) and values Filter objects
    """

    start_time = time.time()
    timestamp = strftime("%Y-%m-%d_%H-%M-%S")

    dest = get_unique_folder_name(f'data/brirs/{Path(hrtf_path).stem}/')
    Path(dest).mkdir(parents=True, exist_ok=False)

    source_positions = generate_source_positions(brir_config.source_positions)

    nr_brirs = get_nr_brirs(brir_config.room_configs, source_positions)
    brir_params = generate_BRIR_params(brir_config.room_configs, source_positions)

    try:
        if brir_config.persist_brirs_individually:  # Save BRIRs as individual files
            with Pool(initializer=initializer, initargs=(brir_config.room_configs, hrtf_path)) as pool:
                for training_coords, brir in tqdm(pool.imap_unordered(run_brir_sim, brir_params),
                                                  total=nr_brirs):
                    brir.save(Path(dest / f'brir_{training_coords}.wav'))
        else:  # Save BRIRs as dictionary in one file
            brir_dict = dict()
            with Pool() as pool:
                for result in tqdm(pool.imap_unordered(run_brir_sim, brir_params), total=nr_brirs):
                    brir_dict[result[0]] = result[1]
            pickle.dump(brir_dict, open(dest / 'brir_dict.pkl', 'wb'))
    except Exception as e:
        logger.error(f'An error occurred during BRIR generation:\n{traceback.print_exc()}')
    finally:
        elapsed_time = str(datetime.timedelta(seconds=time.time() - start_time))
        summary = summarize_brir_generation_info(brir_config, hrtf_path, timestamp, elapsed_time, dest)
        logger.info(summary)
        with open(dest / f'_summary_{timestamp}.txt', 'w') as f:
            f.write(summary)


def generate_source_positions(source_positions: SourcePositionsConfig) -> List[SphericalCoordinates]:
    return [SphericalCoordinates(azimuth, elevation) for azimuth, elevation in
            itertools.product(source_positions.azimuths, source_positions.elevations)]


@persistent_cache
def get_nr_brirs(room_configs: List[RoomConfig], source_positions: List[SphericalCoordinates]) -> int:
    return sum(1 for _ in generate_BRIR_params(room_configs, source_positions))


def summarize_brir_generation_info(brir_config: BRIRConfig, hrtf_path: str,
                                   timestamp: str, elapsed_time: str, dest: Path) -> str:
    """
    Summarizes the BRIR generation info in a human-readable format.
    """

    # Re-generate stuff here so this function stays independent of the rest of the code
    source_positions = generate_source_positions(brir_config.source_positions)
    nr_brirs = get_nr_brirs(brir_config.room_configs, source_positions)
    num_listener_positions = sum(
        len(calculate_listener_positions(room_config.width, room_config.length)) for room_config in
        brir_config.room_configs)
    num_source_positions = len(list(source_positions))

    summary = f'##### BRIR GENERATION INFO #####\n' \
              f'HRTF: {hrtf_path}\n' \
              f'Timestamp: {timestamp}\n' \
              f'Total elapsed time: {elapsed_time}\n' \
              f'Number of Listener Positions: {num_listener_positions}\n' \
              f'Number of Source positions per Listener position: {num_source_positions}\n' \
              f'{len(list(dest.glob("brir_*")))}/{nr_brirs} BRIRs generated\n' \
              f'Config:\n{pprint.pformat(brir_config)}\n\n' \
              f'################################\n' \
              f'Results saved to: {dest}\n' \
              f'################################\n'
    return summary


def generate_BRIR_params(room_configs: List[RoomConfig], source_positions: List[SphericalCoordinates]) -> \
        Generator[TrainingCoordinates, None, None]:
    """
    Generates the parameters for the BRIRs to be computed.
    Listener positions depend on the rooms given.

    Returns:
        Generator of TrainingCoordinates objects
    """

    for room_config in room_configs:
        listener_positions = calculate_listener_positions(room_config.width, room_config.length)
        for listener_position, source_position in itertools.product(listener_positions, source_positions):
            yield TrainingCoordinates(room_config.id, listener_position, source_position)


def calculate_listener_positions(room_width: float, room_length: float) -> List[CartesianCoordinates]:
    """
    Given the size of a room, generates a list of listener positions with the following constraints:
    1. The listener is placed 1.4m from the walls
    2. The listener is placed on a grid with 1m spacing
    3. The grid is centered in the room

    Args:
        room_width: Width of the room in meters
        room_length: Length of the room in meters

    Returns:
        List of CartesianCoordinates containing the listener positions.
    """
    # 1.4m from each wall, 1m between positions, add 1 for the last position
    nr_positions_x = floor(room_width - 2.8) + 1
    nr_positions_y = floor(room_length - 2.8) + 1

    positions_x_start = (room_width - (nr_positions_x - 1)) / 2
    positions_y_start = (room_length - (nr_positions_y - 1)) / 2

    positions = []
    for x in range(nr_positions_x):
        for y in range(nr_positions_y):
            positions.append(CartesianCoordinates(positions_x_start + x, positions_y_start + y))
    return positions


def initializer(room_configs: List[RoomConfig], hrtf_path: str) -> None:
    """
    Initializer function for the multiprocessing Pool to pass the room parameters and HRTF to the worker functions.
    """
    global _room_params
    global _hrtf
    _room_params = {room.id: [room.width, room.length, room.height] for room in room_configs}
    if hrtf_path == 'slab_kemar':
        _hrtf = slab.HRTF.kemar()
    else:
        _hrtf = slab.HRTF(hrtf_path)


def run_brir_sim(brir_params: TrainingCoordinates[int, CartesianCoordinates, SphericalCoordinates]) \
        -> Tuple[TrainingCoordinates, slab.Filter]:
    """
    Wrapper function to calculate the BRIRs for a given set of room configurations and BRIR parameters.
    TODO: Add absorption and wall_filter

    Args:
        brir_params: TrainingCoordinates object containing the room ID, listener position and source position
    Returns:
        Tuple of TrainingCoordinates and slab.Filter
    """
    global _room_params
    global _hrtf
    room_id, listener_position, source_position = brir_params
    # slab.Room takes vertical polar (i.e. spherical) coordinates -> 0 to 359 clockwise
    # No! Seems to take 0 to 359 counter-clockwise.

    room = slab.Room(size=_room_params[room_id], listener=[*listener_position, 1.4],
                     source=[abs(source_position.azim - 360), source_position.elev, 1.4])
    # Trim prob not necessary; is already below length of samples (2s); trying to trim to 2s results in crash
    return TrainingCoordinates(room_id, listener_position, source_position), room.hrir(hrtf=_hrtf).resample(48000)


if __name__ == "__main__":
    main()
