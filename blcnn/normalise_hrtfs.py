import time

import numpy as np
import slab


# Load .sofa files
# Normalize HRTFs -> one scaling factor for all HRTFS

def rms_global(hrtf: slab.HRTF):
    """
    hrtf: slab.HRTF object
    returns scalar RMS across all directions and both channels
    """
    all_data = []
    for ir in hrtf.data:
        all_data.append(ir.data)  # shape (N,2)
    all_data = np.vstack(all_data)  # shape (num_directions*N, 2)
    return rms_stereo(all_data)

def rms_stereo(y):
    """
    y: array shape (T,2) or (2,T)
    returns scalar RMS across both channels
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[0] == 2:
        # channels-first (2, T)
        power = np.mean(y[0]**2 + y[1]**2) / 2.0
    elif y.ndim == 2 and y.shape[1] == 2:
        # samples-first (T, 2)
        power = np.mean(y[:,0]**2 + y[:,1]**2) / 2.0
    else:
        # mono fallback
        power = np.mean(y**2)
    return float(np.sqrt(power))


def normalise_hrtf(hrtf: slab.HRTF, rms_target: float=0.1):
    """Normalise the HRTF object to a target RMS value across all directions and both channels.

    Args:
        hrtf: slab.HRTF object to normalise.
        rms_target: Target RMS value.

    Returns:
        Normalised slab.HRTF object.
    """
    current_rms = rms_global(hrtf)
    scaling_factor = rms_target / current_rms
    for ir in hrtf.data:
        ir.data *= scaling_factor
    return hrtf

if __name__ == '__main__':
    # Normalise all .sofa files in the 'data/hrtfs/' directory
    import os
    hrtf_dir = 'data/hrtfs/'
    for filename in os.listdir(hrtf_dir):
        if filename.endswith('.sofa') and not (filename.endswith('_normalised_kemar.sofa') or filename.endswith('_normalised_1e-1.sofa')):
            filepath = os.path.join(hrtf_dir, filename)
            hrtf = slab.HRTF(filepath)
            original_rms = rms_global(hrtf)
            print(f'Normalising {filename}: Original RMS: {original_rms}')
            rms_target = rms_global(slab.HRTF.kemar())
            print(f'Target RMS (KEMAR): {rms_target}')
            normalised_hrtf = normalise_hrtf(hrtf, rms_target=rms_target)
            print(f'Normalised RMS: {rms_global(normalised_hrtf)}\n')
            normalised_filepath = os.path.join(hrtf_dir, filename[:-5] + '_normalised_kemar.sofa')
            normalised_hrtf.write_sofa(normalised_filepath)
            # Add info to log file
            with open(os.path.join(hrtf_dir, 'normalisation_log.txt'), 'a') as log_file:
                log_file.write(f'{time.strftime("%Y-%m-%d_%H-%M-%S")} --- {filename}: Original RMS: {original_rms}, Target RMS (slab KEMAR): {rms_target}, Normalised RMS: {rms_global(normalised_hrtf)}\n')


    # Same thing, but normalise to RMS of 0.1
    for filename in os.listdir(hrtf_dir):
        if filename.endswith('.sofa') and not (filename.endswith('_normalised_kemar.sofa') or filename.endswith('_normalised_1e-1.sofa')):
            filepath = os.path.join(hrtf_dir, filename)
            hrtf = slab.HRTF(filepath)
            original_rms = rms_global(hrtf)
            print(f'Normalising {filename}: Original RMS: {original_rms}')
            rms_target = 0.1
            print(f'Target RMS (fixed): {rms_target}')
            normalised_hrtf = normalise_hrtf(hrtf, rms_target=0.1)
            print(f'Normalised RMS: {rms_global(normalised_hrtf)}\n')
            normalised_filepath = os.path.join(hrtf_dir, filename[:-5] + '_normalised_1e-1.sofa')
            normalised_hrtf.write_sofa(normalised_filepath)
            # Add info to log file
            with open(os.path.join(hrtf_dir, 'normalisation_log.txt'), 'a') as log_file:
                log_file.write(f'{time.strftime("%Y-%m-%d_%H-%M-%S")} --- {filename}: Original RMS: {original_rms}, Target RMS (fixed): {rms_target}, Normalised RMS: {rms_global(normalised_hrtf)}\n')
