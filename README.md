# BinauralLocalizationCNN
CNN that mimics human auditory source localization behavior based on: [Publication](https://www.nature.com/articles/s41562-021-01244-z#).  
Original GitHub repo can be found [here](https://github.com/afrancl/BinauralLocalizationCNN).
## Setup
### Environments
This project uses _conda_ for its Python environments.  
**Linux**: Install the x86 environment with `conda env create -n blcnn-gpu-env -f blcnn-x86-env.yml`  
**Apple M1**: Install the arm64 environment with `conda env create -n blcnn-osx-arm64-env -f blcnn-osx-arm64-env.yml`

On Linux install libsndfile and PortAudio manually as required by slab: `sudo apt-get install libsndfile1 libportaudio21`

#### Data
Download the model weights from [here](https://www.dropbox.com/sh/af6vaotxt41i7pe/AACfTzMxMLfv-Edmn33S4gTpa?dl=0) and place the `Binaural Localization Net Weights` folder into `models/`.  
Get your favourite HRTFs from [here](https://www.sofaconventions.org/mediawiki/index.php/Files) and place them in `data/hrtfs/`.  
Get your favourite stimulus sounds and place the folder containing them into `data/raw/`.

#### Transfer model weights to TF2/Keras
The original model was trained in TF1. This project ported the models to TensorFlow 2 / Keras.
To convert the weights to TF2/Keras, run `python convert_models_to_tf2.py` after placing the weights in the correct folder.

## Usage
The configuration for the scripts is managed through a YAML file (`config.yml`). This file contains various settings and parameters required for the execution of the scripts. The `util.py` file provides functions to load and parse this configuration file.

The main variable that changes between runs is the HRTF that's used.
In each step, the HRTFs to be used can be specified individually.
For the first step (generate BRIRs), the HRTFs are specified through their path.
In subsequent steps, the HRTFs are specified through their label, which is the name of the HRTF file without the extension.

The scripts save a summary of the execution along with a copy of the config to avoid confusion between runs.

### `generate_brirs.py`
This script generates Binaural Room Impulse Responses (BRIRs) needed to simulate the auditory environment.
- `hrtfs`: List of paths to HRTFs to generate BRIRs for.
- `source_positions`: List of source positions to generate BRIRs for; defaults taken from McDermott's paper.
- `room_configs`: List of room configurations, defaults taken from McDermott's paper.
- `persist_brirs_individually`: Boolean to determine if BRIRs should be saved as individual files (yes they probably should).

### `generate_cochleagrams.py`
This script generates cochleagrams from audio stimuli using the specified HRTFs.
- `hrtf_labels`: List of HRTF labels to use.
- `stim_path`: Path to the directory containing the audio stimuli.
- `bkgd_path`: Path to the directory containing background sounds (not used at the moment).
- `use_bkgd`: Boolean to determine if background sounds should be used (set to False for now).

### `run_models.py`
This script runs the trained models on the generated cochleagrams to perform auditory source localization.
- `hrtf_labels`: List of HRTF labels to use.
- `models_to_use`: List of models to use for the predictions; IDs must be from 1 to 10.

### `plotting.py`
This script generates plots and visualizations from the model predictions.
- `hrtf_labels`: List of HRTF labels to use.
- `binned`: Boolean to determine if the predictions should be binned.
- `nr_elevation_bins`: Number of bins to use for the elevation.
- `nr_azimuth_bins`: Number of bins to use for the azimuth.
- `show_single_responses`: Boolean to determine if the single responses should be shown.
- `style`: Preset style of the plots ('debug', 'hofman', or None).

### Other scripts
- `inspect_tfrecord.py`: Helper tool to look at the insides of a TFRecord file.
- `inv_coch.py`: Inverse cochleagram transform to hear the sounds from the cochleagrams.
- `mem_usage.py`: Script to calculate the potential memory usage of the models.
- `net_builder.py`: Contains the code to build the model architectures from the Tensorflow 1 config arrays, as well as the parser for the samples from the tfrecord files (... which really should be in a different file)
- `persistent_cache.py`: A simple persistent cache decorator.
- `util.py`: Contains utility functions, e.g., to load and parse the configuration file.

# Links
- 2-sec sounds: https://mcdermottlab.mit.edu/svnh/Natural-Sound/Stimuli.html (currently using different stim set)
- 7-textures, find under https://mcdermottlab.mit.edu/downloads.html


