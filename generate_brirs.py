import itertools
import logging
import pickle
from math import floor
from multiprocessing import Pool
from pathlib import Path
from time import strftime
from typing import NamedTuple, List, Dict, Generator, Tuple

import slab
from tqdm import tqdm

logger = logging.getLogger(__name__)

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


MCDERMOTT_SOURCE_POSITIONS = [SphericalCoordinates(azimuth, elevation)
                              for azimuth, elevation in itertools.product(range(-180, 180, 5), range(0, 61, 10))]
MCDERMOTT_ROOM_CONFIGS = {1: RoomConfig(RoomSize(9, 9, 10), [0.1, ]),  # TODO: Add real absorption coefficients
                          2: RoomConfig(RoomSize(4, 5, 3), [0.1, ]),  # height from 2m to 3m bc src out of bounds
                          3: RoomConfig(RoomSize(10, 10, 4), [0.1, ]),
                          4: RoomConfig(RoomSize(5, 8, 5), [0.1, ]),
                          5: RoomConfig(RoomSize(4, 4, 4), [0.1, ])}  # width, length from 3m to 4m bc src out of bounds


def now(): return strftime("%Y-%m-%d_%H-%M-%S")


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
    return TrainingCoordinates(room_id, listener_position, source_position), room.hrir().resample(48000)


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

    log_brir_generation_info(room_configs)

    nr_brirs = sum(1 for _ in generate_BRIR_params(room_configs, MCDERMOTT_SOURCE_POSITIONS))
    brir_params = generate_BRIR_params(room_configs, MCDERMOTT_SOURCE_POSITIONS)
    timestamp = now()

    if persist_brirs_individually:
        Path(f'data/brirs_{timestamp}/').mkdir(parents=True, exist_ok=True)
        with Pool() as pool:
            for result in tqdm(pool.imap_unordered(run_brir_sim, brir_params), total=nr_brirs):
                # Persist the BRIRs individually
                result[1].save(Path('data', f'brirs_{timestamp}', f'brir_{result[0]}.wav'))
                # pickle.dump(result[1].resample(48000), open(f'data/brirs_{timestamp}/brir_{result[0]}.pkl', 'wb'))

                # Resample 48 -> 35min, 502kB
                # no resample -> 20min, 460kB
                # Pickle.dump resample 48 -> 35min, 502kB
                # pickle.dump -> 16min, 460kB
                # difference of 3GB. 33GB vs 36GB, ok for 15 minutes speed improvement later

    else:
        Path(f'data/brirs{timestamp}/').mkdir(parents=True, exist_ok=True)
        brir_dict = dict()
        with Pool() as pool:
            for result in tqdm(pool.imap_unordered(run_brir_sim, brir_params), total=nr_brirs):
                # Add the BRIRs to the dictionary and persist later
                brir_dict[result[0]] = result[1]
        pickle.dump(brir_dict, open(f'data/brirs_{timestamp}/brir_dict_{now()}.pkl', 'wb'))


def log_brir_generation_info(room_configs: Dict[int, RoomConfig]):
    """
    Given a list of room configurations, prints the number of listener positions and the number of BRIRs to generate in total.
    """
    nr_brirs = sum(1 for _ in generate_BRIR_params(room_configs, MCDERMOTT_SOURCE_POSITIONS))
    num_listener_positions = sum(
        len(calculate_listener_positions(room_config.room_size)) for room_id, room_config in room_configs.items())
    num_source_positions = len(list(MCDERMOTT_SOURCE_POSITIONS))

    logger.info('##### DATA GENERATION INFO #####')
    logger.info(f'Number of Rooms: {len(room_configs.items())}')
    logger.info(f'Number of Listener Positions: {num_listener_positions}')
    logger.info(f'Number of Source positions per Listener position: {num_source_positions}')
    logger.info(f'Generating {nr_brirs} BRIRs')


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
