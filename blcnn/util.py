from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Any

import yaml

'''
Structure:
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
        stim_path: str
        bkgd_path: str
        use_bkgd: bool
    model_playground: ModelPlaygroundConfig
        hrtf_labels: List[str]
        model_path: str
        models_to_use: List[int]
'''


@dataclass
class RangeConfig:
    start: int
    stop: int
    step: int


@dataclass
class SourcePositionsConfig:
    azimuth: RangeConfig
    elevation: RangeConfig


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
    stim_path: str
    bkgd_path: str
    use_bkgd: bool


@dataclass
class RunModelsConfig:
    hrtf_labels: List[str]
    models_to_use: List[int]

@dataclass
class PlottingConfig:
    hrtf_labels: List[str]
    binned: bool
    nr_elevation_bins: int
    nr_azimuth_bins: int
    show_single_responses: bool
    style: str


@dataclass
class Config:
    generate_brirs: BRIRConfig
    generate_cochleagrams: CochleagramConfig
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
                azimuth=RangeConfig(**raw_config['generate_brirs']['source_positions']['azimuth']),
                elevation=RangeConfig(**raw_config['generate_brirs']['source_positions']['elevation'])
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
            stim_path=raw_config['generate_cochleagrams']['stim_path'],
            bkgd_path=raw_config['generate_cochleagrams']['bkgd_path'],
            use_bkgd=raw_config['generate_cochleagrams']['use_bkgd']
        ),
        run_models=RunModelsConfig(
            hrtf_labels=raw_config['run_models']['hrtf_labels'],
            models_to_use=raw_config['run_models']['models_to_use']
        ),
        plotting=PlottingConfig(
            hrtf_labels=raw_config['plotting']['hrtf_labels'],
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
