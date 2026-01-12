from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Any

import yaml
import slab

'''
Structure (old):
Config
    generate_brirs: BRIRConfig
        hrtfs: List[str]
        source_positions: SourcePositionsConfig
            azimuth: RangeConfig
                start: int
                stop: int
                step: int
            elevation: RangeConfig
                start: int
                stop: int
                step: int
        room_configs: List[RoomConfig]
            room_id: int
            width: float
            length: float
            height: float
        persist_brirs_individually: bool
    generate_cochleagrams: CochleagramConfig
        hrtf_labels: List[str]
        stim_paths: List[str]
        bkgd_path: str
        use_bkgd: bool
    model_playground: ModelPlaygroundConfig
        hrtf_labels: List[str]
        model_path: str
        models_to_use: List[int]
'''


@dataclass
class SourcePositionsConfig:
    azimuths: List[int]
    elevations: List[int]


@dataclass
class RoomConfig:
    id: int
    width: float
    length: float
    height: float

    # sort by room_id, may be unnecessary
    def __lt__(self, other):
        return self.id < other.id


@dataclass
class BRIRConfig:
    hrtfs: List[str]
    source_positions: SourcePositionsConfig
    room_configs: List[RoomConfig]
    persist_brirs_individually: bool


@dataclass
class CochleagramConfig:
    hrtf_labels: List[str]
    stim_paths: List[str]
    source_positions: SourcePositionsConfig
    bkgd_path: str
    use_bkgd: bool
    anechoic: bool
    train_test_split: float
    generation_base_probability: float


@dataclass
class FreezeTrainingConfig:
    labels: List[str]
    models_to_use: List[int]
    ngrams: List[int]


@dataclass
class RunModelsConfig:
    folder: str
    labels: List[str]
    models_to_use: List[int]


@dataclass
class PlottingConfig:
    labels: List[str]
    data_selection: str
    folded: bool
    binned: bool
    nr_elevation_bins: int
    nr_azimuth_bins: int
    show_single_responses: bool
    style: str


@dataclass
class Config:
    generate_brirs: BRIRConfig
    generate_cochleagrams: CochleagramConfig
    freeze_training: FreezeTrainingConfig
    run_models: RunModelsConfig
    plotting: PlottingConfig


def load_config(file_path: str) -> Config:
    """
    Load the configuration from the given YAML file.
    Args:
        file_path: The path to the YAML file.

    Returns:
        The configuration data class.
    """
    with open(file_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    # Map the nested YAML dictionary to the data classes
    return Config(
        generate_brirs=BRIRConfig(
            hrtfs=raw_config['generate_brirs']['hrtfs'],
            source_positions=SourcePositionsConfig(
                azimuths=raw_config['generate_brirs']['source_positions']['azimuths'],
                elevations=raw_config['generate_brirs']['source_positions']['elevations']
            ),
            room_configs=[
                RoomConfig(
                    id=room['id'],
                    width=room['width'],
                    length=room['length'],
                    height=room['height']
                ) for room in raw_config['generate_brirs']['room_configs']
            ],
            persist_brirs_individually=raw_config['generate_brirs']['persist_brirs_individually']
        ),
        generate_cochleagrams=CochleagramConfig(
            hrtf_labels=raw_config['generate_cochleagrams']['hrtf_labels'],
            stim_paths=raw_config['generate_cochleagrams']['stim_paths'],
            source_positions=SourcePositionsConfig(
                azimuths=raw_config['generate_cochleagrams']['source_positions']['azimuths'],
                elevations=raw_config['generate_cochleagrams']['source_positions']['elevations']
            ),
            bkgd_path=raw_config['generate_cochleagrams']['bkgd_path'],
            use_bkgd=raw_config['generate_cochleagrams']['use_bkgd'],
            anechoic=raw_config['generate_cochleagrams']['anechoic'],
            train_test_split=raw_config['generate_cochleagrams']['train_test_split'],
            generation_base_probability= raw_config['generate_cochleagrams']['generation_base_probability']
        ),
        freeze_training=FreezeTrainingConfig(
            labels=raw_config['freeze_training']['labels'],
            models_to_use=raw_config['freeze_training']['models_to_use'],
            ngrams=raw_config['freeze_training']['ngrams']
        ),
        run_models=RunModelsConfig(
            folder=raw_config['run_models']['folder'],
            labels=raw_config['run_models']['labels'],
            models_to_use=raw_config['run_models']['models_to_use']
        ),
        plotting=PlottingConfig(
            labels=raw_config['plotting']['labels'],
            data_selection=raw_config['plotting']['data_selection'],
            folded=raw_config['plotting']['folded'],
            binned=raw_config['plotting']['binned'],
            nr_elevation_bins=raw_config['plotting']['nr_elevation_bins'],
            nr_azimuth_bins=raw_config['plotting']['nr_azimuth_bins'],
            show_single_responses=raw_config['plotting']['show_single_responses'],
            style=raw_config['plotting']['style']
        )
    )


def get_unique_folder_name(base_name):
    """
    Generate a unique folder name. Use the base name if it doesn't exist,
    otherwise append a numeric suffix to ensure uniqueness.
    """
    base_path = Path(base_name)
    if not base_path.exists():
        return base_path  # Return the base name directly if it doesn't exist

    counter = 1
    while (folder_path := base_path.with_name(f"{base_path.stem}_{counter}")).exists():
        counter += 1
    return folder_path


def CNNpos_to_loc(CNN_pos):
    """
    convert bin label in the CNN from Francl 2022 into [azim, elev] positions
    :param CNN_pos: int, [0, 503]
    :return: tuple, (azi, ele)

    # Old conversion to interaural polar coordinates, but might be
    # wrong if original vertical polar coords use ccw wrapping
    if azim >= 180:
        azim -= 360
    """
    div, mod = divmod(CNN_pos, 72)
    azim = mod * 5
    elev = div * 10
    return azim, elev

def loc_to_CNNpos(azim, elev):
    """
    convert [azim, elev] positions into bin label in the CNN from Francl 2022
    :param azim: int, [0, 355]
    :param elev: int, [0, 60]
    :return: int, [0, 503]
    """
    azim = azim % 360  # wrap around
    div = elev // 10
    mod = azim // 5
    return div * 72 + mod
