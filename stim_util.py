from pycochleagram import utils as utl
import slab
import numpy as np
from nnresample import resample


def hanning_window(stim, ramp_duration_ms, SAMPLERATE=48000):
    stim_np = np.array(stim)
    stim_dur_smp = stim_np.shape[0]
    ramp_dur_smp = int(np.floor(ramp_duration_ms*SAMPLERATE/1000))
    hanning_window = np.hanning(ramp_dur_smp*2)
    onset_win = stim_np[:ramp_dur_smp] * hanning_window[:ramp_dur_smp]
    middle = stim_np[ramp_dur_smp:stim_dur_smp-ramp_dur_smp]
    end_win = stim_np[stim_dur_smp-ramp_dur_smp:] * hanning_window[ramp_dur_smp:]
    windowed_stim = np.concatenate((onset_win, middle, end_win))
    return windowed_stim


# stim need to be standardized before feeding into cochleagram
# TODO: in the model, the cochleagram is generated at 48000 Hz
def normalize_binaural_stim(orig_stim, orig_sr, target_sr=48000, scaling_max=0.1, min_len=2):
    # TODO: according to real_world_audio_rescale, after rescaling the values are in [-0.1, 0.1]
    # TODO: thus the scaling_max is set to 0.1 by default
    """
    read binaural sound from wavefile and prepare it for feeding into cochleagram wrapper
    Args:
        orig_stim: N-by-2 np array, binaural sound
        orig_sr: int, original sampling rate
        target_sr: int, resulting sampling frequency
        scaling_max: float, maximum value of resulting stim
        min_len: float, minimum length of the stimulus, second

    Returns:
        standardized binaural stim, as well as sampling frequency of the stim
    """
    assert orig_stim.shape[1] == 2, 'a binaural stimulus with shape N-by-2 is needed'
    assert orig_stim.shape[0] >= orig_sr * min_len, 'the stimulus must have at least {} ' \
                                                    'seconds duration'.format(min_len)
    stim_wav = scaling_max * utl.rescale_sound(orig_stim, 'normalize')
    stim_wav = stim_wav.T
    if orig_sr != target_sr:
        # stim_wav_empty = np.empty_like(stim_wav)
        # print("resampling")
        stim_wav_l = resample(stim_wav[0], target_sr, orig_sr, As=75, N=64001)
        stim_wav_r = resample(stim_wav[1], target_sr, orig_sr, As=75, N=64001)
        stim_freq = target_sr
        stim_wav = np.vstack([stim_wav_l, stim_wav_r])
    else:
        stim_freq = orig_sr
    return stim_wav, stim_freq


def CNNpos_to_loc(CNN_pos, bin_size=5):
    """
    convert bin label in the CNN from Francl 2022 into [azim, elev] positions
    :param CNN_pos: int, [0, 503]
    :param bin_size: int, degree. note that elevation bin size is 2*bin_size
    :return: tuple, (azi, ele)
    """
    n_azim = int(360 / bin_size)
    bin_idx = divmod(CNN_pos, n_azim)

    return bin_size * bin_idx[1], bin_size * 2 * bin_idx[0]


# NOTE: for the position bins in the CNN, azimuth is [0, 360) and elevation is [0, 60]
# azimuth is binned with 5 degree bins with int(azim / bin) and elevation with 10 degree bin
# label is calculated as 72 * elev_idx + azim_idx
def loc_to_CNNpos(azim, elev, elev_range=(0, 60), bin_size=5):
    """
    convert [azim, elev] positions into bin label in the CNN from Francl 2022
    :param azim: int, azimuth angle, 0 degree is straight ahead
    :param elev: int, elevation angle, should be in [0, 60]
    :param elev_range: tuple, valid elevation angle range
    :param bin_size: int, degree. note that elevation bin size is 2*bin_size
    :return: int, bin label
    """
    # flip the negative angles
    if azim < 0:
        azim = 360 + azim

    if elev < elev_range[0] or elev > elev_range[1]:
        raise ValueError("elevation angle: {} out of valid range: {}".
                         format(elev, elev_range))
    # TODO: maybe multiply by 36 instead of 72?
    return int(elev / 2 / bin_size) * int(360 / bin_size) + int(azim / bin_size)


def zero_padding(stim, type="front", goal_duration=2.1):
    if not isinstance(stim, slab.Sound):
        raise ValueError("stimulus must be instance of slab.Sound!")
    curr_length_ns = stim.n_samples
    if type == "frontback":
        missing_length_ns = int((goal_duration * stim.samplerate - curr_length_ns) / 2)
        padding = slab.Sound.silence(missing_length_ns, stim.samplerate, stim.n_channels)
        return slab.Sound.sequence(padding, stim, padding)
    elif type == "front":
        missing_length_ns = int((goal_duration * stim.samplerate - curr_length_ns))
        padding = slab.Sound.silence(missing_length_ns, stim.samplerate, stim.n_channels)
        return slab.Sound.sequence(padding, stim)
    elif type == "back":
        missing_length_ns = int((goal_duration * stim.samplerate - curr_length_ns))
        padding = slab.Sound.silence(missing_length_ns, stim.samplerate, stim.n_channels)
        return slab.Sound.sequence(stim, padding)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("TkAgg")
    sound = slab.Sound.whitenoise(1.0)
    goal_duration = 2.1
    front_padded = zero_padding(stim=sound, type="front", goal_duration=goal_duration)
    back_padded = zero_padding(stim=sound, type="back", goal_duration=goal_duration)
    front_back_padded = zero_padding(stim=sound, type="frontback", goal_duration=goal_duration)
    fig, ax = plt.subplots(2, 2)
    plt.tight_layout()
    sound.waveform(axis=ax[0][0])
    front_padded.waveform(axis=ax[0][1])
    back_padded.waveform(axis=ax[1][0])
    front_back_padded.waveform(axis=ax[1][1])
