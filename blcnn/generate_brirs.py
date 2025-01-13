import itertools
import logging
import pickle
import sys
import time
from math import floor
from multiprocessing import Pool
from pathlib import Path
from time import strftime
from typing import NamedTuple, List, Dict, Generator, Tuple

import coloredlogs
import slab
from tqdm import tqdm

from util import get_unique_folder_name

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


class RoomSize(NamedTuple):
    width: float
    length: float
    height: float


class RoomConfig(NamedTuple):
    room_size: RoomSize
    absorption: List[float]


class TrainingCoordinates(NamedTuple):
    room_id: int
    listener_position: CartesianCoordinates
    source_position: SphericalCoordinates

    def __str__(self):
        return f'{self.room_id}r_{self.listener_position.x}x_{self.listener_position.y}y_{self.source_position.elev}e_{self.source_position.azim}a'


#
# MCDERMOTT_SOURCE_POSITIONS = [SphericalCoordinates(azimuth, elevation)
#                               for azimuth, elevation in itertools.product(range(-180, 180, 5), range(0, 61, 10))]

# vertical_polar coordinates from 0 to 355 counterclockwise
MCDERMOTT_SOURCE_POSITIONS = [SphericalCoordinates(azimuth, elevation)
                              for azimuth, elevation in itertools.product(range(0, 360, 5), range(0, 61, 10))]
MCDERMOTT_ROOM_CONFIGS = {1: RoomConfig(RoomSize(9, 9, 10), [0.1, ]),  # TODO: Add real absorption coefficients
                          2: RoomConfig(RoomSize(4, 5, 3), [0.1, ]),  # height from 2m to 3m bc src out of bounds
                          3: RoomConfig(RoomSize(10, 10, 4), [0.1, ]),
                          4: RoomConfig(RoomSize(5, 8, 5), [0.1, ]),
                          5: RoomConfig(RoomSize(4, 4, 4), [0.1, ])}  # width, length from 3m to 4m bc src out of bounds

# CUSTOM_HRTF_FILENAME = 'hrtf_b_nh2.sofa'
# CUSTOM_HRTF = slab.HRTF(Path('../data/hrtfs', CUSTOM_HRTF_FILENAME))
CUSTOM_HRTF_FILENAME = 'slab_default_kemar'
CUSTOM_HRTF = slab.HRTF.kemar()

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
    generate_and_persist_BRIRs(MCDERMOTT_ROOM_CONFIGS)


def generate_and_persist_BRIRs(room_configs: Dict[int, RoomConfig], persist_brirs_individually: bool = True) -> None:
    """
    Given a list of room configurations, generates the corresponding BRIRs and stores them in a dictionary.
    The BRIRs can be accessed using (room_id, listener_position, source_position).
    Part of reimplementation of McDermott's data generation pipeline.

    Note: I think HRIR here is really a BRIR. Assuming this for now. Technically, it's a Filter object in slab.
    Note: 64 bytes for one room.hrir() Filter object.
    Note: Dict keeps the correct filters.

    Args:
        Dict[int, RoomConfig]: Dict of {room_id: RoomConfig} containing the room size and absorption coefficients

    Returns:
        Dictionary with keys (room_id, listener_position, source_position) and values Filter objects
    """

    timestamp = strftime("%Y-%m-%d_%H-%M-%S")

    summary = summarize_brir_generation_info(room_configs, timestamp, persist_brirs_individually)
    logger.info(summary)

    dest = get_unique_folder_name(f'data/brirs/{CUSTOM_HRTF_FILENAME.split(".")[0]}/')

    Path(dest).mkdir(parents=True, exist_ok=False)
    with open(dest / f'_summary_{timestamp}.txt', 'w') as f:
        f.write(summary)

    nr_brirs = sum(1 for _ in generate_BRIR_params(room_configs, MCDERMOTT_SOURCE_POSITIONS))
    brir_params = generate_BRIR_params(room_configs, MCDERMOTT_SOURCE_POSITIONS)

    # sys.exit()

    if persist_brirs_individually:  # Save BRIRs as individual files
        with Pool() as pool:
            for training_coords, brir in tqdm(pool.imap_unordered(run_brir_sim, brir_params), total=nr_brirs):
                brir.save(Path(dest / f'brir_{training_coords}.wav'))
    else:  # Save BRIRs as dictionary in one file
        brir_dict = dict()
        with Pool() as pool:
            for result in tqdm(pool.imap_unordered(run_brir_sim, brir_params), total=nr_brirs):
                brir_dict[result[0]] = result[1]
        pickle.dump(brir_dict, open(dest / 'brir_dict.pkl', 'wb'))


def summarize_brir_generation_info(room_configs: Dict[int, RoomConfig], timestamp: str, persist_brirs_individually: bool) -> str:
    """
    Given a list of room configurations, prints the number of listener positions and the number of BRIRs to generate in total.
    """
    nr_brirs = sum(1 for _ in generate_BRIR_params(room_configs, MCDERMOTT_SOURCE_POSITIONS))
    num_listener_positions = sum(
        len(calculate_listener_positions(room_config.room_size)) for room_id, room_config in room_configs.items())
    num_source_positions = len(list(MCDERMOTT_SOURCE_POSITIONS))


    summary = f'##### BRIR GENERATION INFO #####\n' \
               f'Timestamp: {timestamp}\n' \
               f'BRIRs as individual files: {persist_brirs_individually}\n' \
               f'HRTF: {CUSTOM_HRTF_FILENAME}\n' \
               f'HRTF samplerate: {CUSTOM_HRTF.samplerate}\n\n' \
               f'Room Configs: {room_configs}\n' \
               f'Number of Rooms: {len(room_configs.items())}\n' \
               f'Number of Listener Positions: {num_listener_positions}\n' \
               f'Number of Source positions per Listener position: {num_source_positions}\n' \
               f'Generating {nr_brirs} BRIRs\n'
    return summary


def generate_BRIR_params(room_configs: Dict[int, RoomConfig], source_positions: List[SphericalCoordinates]) -> \
        Generator[TrainingCoordinates, None, None]:
    """
    Generates the parameters for the BRIRs to be computed.
    Listener positions depend on the rooms given.

    Returns:
        Generator of TrainingCoordinates objects
    """

    for room_id, room_config in room_configs.items():
        listener_positions = calculate_listener_positions(room_config.room_size)
        for listener_position, source_position in itertools.product(listener_positions, source_positions):
            yield TrainingCoordinates(room_id, listener_position, source_position)


def calculate_listener_positions(room_size: RoomSize) -> List[CartesianCoordinates]:
    """
    Given the size of a room, generates a list of listener positions with the following constraints:
    1. The listener is placed 1.4m from the walls
    2. The listener is placed on a grid with 1m spacing
    3. The grid is centered in the room

    Args:
        room_size: NamedTuple with the size of the room RoomSize(width, length, height)

    Returns:
        List of CartesianCoordinates containing the listener positions.
    """
    # 1.4m from each wall, 1m between positions, add 1 for the last position
    nr_positions_x = floor(room_size.width - 2.8) + 1
    nr_positions_y = floor(room_size.length - 2.8) + 1

    positions_x_start = (room_size.width - (nr_positions_x - 1)) / 2
    positions_y_start = (room_size.length - (nr_positions_y - 1)) / 2

    positions = []
    for x in range(nr_positions_x):
        for y in range(nr_positions_y):
            positions.append(CartesianCoordinates(positions_x_start + x, positions_y_start + y))
    return positions


def run_brir_sim(brir_params: TrainingCoordinates[int, CartesianCoordinates, SphericalCoordinates]) -> Tuple[
    TrainingCoordinates, slab.Filter]:
    """
    Wrapper function to calculate the BRIRs for a given set of room configurations and BRIR parameters.
    TODO: Add absorption and wall_filter
    TODO: Somehow pass room_config to function instead of global variable

    Args:
        brir_params: TrainingCoordinates object containing the room ID, listener position and source position
    Returns:
        Tuple of TrainingCoordinates and slab.Filter
    """
    room_id, listener_position, source_position = brir_params
    room = slab.Room(size=MCDERMOTT_ROOM_CONFIGS[room_id].room_size, listener=[*listener_position, 1.4],
                     source=[*source_position, 1.4])
    # Trim prob not necessary; is already below length of samples (2s); trying to trim to 2s results in crash
    return TrainingCoordinates(room_id, listener_position, source_position), room.hrir(hrtf=CUSTOM_HRTF).resample(48000)


if __name__ == "__main__":
    main()
