"""
Contains copies of legacy functions that are used in the new codebase.
"""
import sys
import os
import numpy as np

# Add 'pycochleagram' to the Python path
pycochleagram_path = os.path.abspath('./pycochleagram')
if pycochleagram_path not in sys.path:
    sys.path.append(pycochleagram_path)

from pycochleagram import cochleagram as cgm
from pycochleagram import utils as utl
import slab
from nnresample import resample


# cochleagram
# NOTE: parameters signal_rate, coch_freq_lims, final_stim_length should NOT be changed to match the required model
#  inputs
# TODO: removing ILD/ITD should be before the cochleagram
def cochleagram_wrapper(stim: np.ndarray, sig_samplerate=48000,
                        coch_gen_sig_cutoff=2, coch_freq_lims=(30, 20000),
                        minimum_padding=0.35, final_stim_length=1,
                        hanning_windowed=True, sliced=True, dual_channel=True):
    """
    pass the stimulus, stim through the cochleagram
    Args:
        stim: N-by-2 np array
        sig_samplerate: stimulus sampling rate
        coch_gen_sig_cutoff: maximum length of the stimulus to be used, second
        coch_freq_lims: 2 elements tuple/list/array, low and high limits of cochlea frequency
        minimum_padding: starting parts of the stimulus to be ignored, second
        final_stim_length: duration of the result, second
        hanning_windowed: if a hanning window is to be used to window the stimulus
        sliced: if pick a random `final_stim_length` duration portion of the stimulus
        dual_channel: if the L/R channels in the resulting cochleagram should be stacked.
            if stacked, the result will be 36-N-2, otherwise 72-N

    Returns:
        resulting cochleagram, np array
    """
    # time to n samples conversion
    final_stim_length_n = round(final_stim_length * sig_samplerate)
    minimum_padding_n = round(minimum_padding * sig_samplerate)
    coch_gen_sig_cutoff_n = round(coch_gen_sig_cutoff * sig_samplerate)

    stim_freq = sig_samplerate

    # Trim signal
    stim = stim[:, :coch_gen_sig_cutoff_n]
    # delay = 15000
    # first dimension due to transpose
    sig_length_in_samples = stim.shape[1]
    # sample_factor is a Positive integer that determines how densely ERB function will be sampled to
    # create bandpass filters. see pycochleagram for more details
    sample_factor = 1

    # Apply a hanning window to the stimulus
    if hanning_windowed:
        r_channel = apply_hanning_window(stim[1], 20, sample_rate=44100)
        l_channel = apply_hanning_window(stim[0], 20, sample_rate=44100)
    else:
        r_channel = stim[1]
        l_channel = stim[0]

    # calculate subbands
    # -> Calls the cochleagram function from pycochleagram, don't inspect for now, but profile
    # Apparently can be run in batched mode, batch dimension is the first dimension -> Otherwise creates redundant filters
    # Maybe a @lru_cache decorator can be used to cache the filters?
    # -> Returns np.array of shape (num_channels, num_samples)
    subbands_r = cgm.human_cochleagram(r_channel, stim_freq, low_lim=coch_freq_lims[0], hi_lim=coch_freq_lims[1],
                                       sample_factor=sample_factor, padding_size=10000,
                                       ret_mode='subband').astype(np.float32)
    subbands_l = cgm.human_cochleagram(l_channel, stim_freq, low_lim=coch_freq_lims[0], hi_lim=coch_freq_lims[1],
                                       sample_factor=sample_factor, padding_size=10000,
                                       ret_mode='subband').astype(np.float32)

    if sliced:
        front_limit = minimum_padding_n
        back_limit = sig_length_in_samples - minimum_padding_n - final_stim_length_n
        jitter = np.random.randint(round(front_limit), round(back_limit))
        front_slice = jitter
        back_slice = jitter + final_stim_length_n
        # 44100*300ms = 13000
        subbands_l = subbands_l[:, front_slice:back_slice]
        subbands_r = subbands_r[:, front_slice:back_slice]

    if dual_channel:
        num_channels = subbands_l.shape[0] - 2 * sample_factor
        subbands = np.empty([num_channels, final_stim_length_n, 2], dtype=subbands_l.dtype)
        # not taking first and last filters because we don't want the low and
        # highpass filters
        subbands[:, :, 0] = subbands_l[sample_factor:-sample_factor]
        subbands[:, :, 1] = subbands_r[sample_factor:-sample_factor]
    else:
        # Interleaving subbands,so local filters can access both channels
        num_channels = subbands_l.shape[0] - 2 * sample_factor
        subbands = np.empty([(2 * num_channels), final_stim_length_n], dtype=subbands_l.dtype)
        subbands[0::2] = subbands_l[sample_factor:-sample_factor]
        subbands[1::2] = subbands_r[sample_factor:-sample_factor]

    # Cut anything -60 dB below peak
    max_val = subbands.max() if subbands.max() > abs(subbands.min()) else abs(subbands.min())
    cutoff = max_val / 1000
    subbands[np.abs(subbands) < cutoff] = 0
    # text input as bytes so bytes objects necessary for comparison
    return subbands


def apply_hanning_window(stim, ramp_duration_ms, sample_rate=48000):
    """
    Apply hanning window to the stimulus
    Args:
        stim:
        ramp_duration_ms:
        sample_rate:

    Returns:
    """

    stim_np = np.array(stim)
    stim_dur_smp = stim_np.shape[0]
    ramp_dur_smp = int(np.floor(ramp_duration_ms * sample_rate / 1000))
    hanning_window = np.hanning(ramp_dur_smp * 2)
    onset_win = stim_np[:ramp_dur_smp] * hanning_window[:ramp_dur_smp]
    middle = stim_np[ramp_dur_smp:stim_dur_smp - ramp_dur_smp]
    end_win = stim_np[stim_dur_smp - ramp_dur_smp:] * hanning_window[ramp_dur_smp:]
    windowed_stim = np.concatenate((onset_win, middle, end_win))
    return windowed_stim


def zero_padding(stim, type="front", goal_duration=2.1):
    if not isinstance(stim, slab.Sound):
        raise ValueError("stimulus must be instance of slab.Sound!")
    if stim.duration > 2.0:
        stim = stim.trim(0.0, 2.0)
    curr_n_samples = stim.n_samples
    if type == "frontback":
        missing_length_ns = int((goal_duration * stim.samplerate - curr_n_samples) / 2)
        padding = slab.Sound.silence(missing_length_ns, stim.samplerate, stim.n_channels)
        return slab.Sound.sequence(padding, stim, padding)
    elif type == "front":
        missing_length_ns = int((goal_duration * stim.samplerate - curr_n_samples))
        padding = slab.Sound.silence(missing_length_ns, stim.samplerate, stim.n_channels)
        return slab.Sound.sequence(padding, stim)
    elif type == "back":
        missing_length_ns = int((goal_duration * stim.samplerate - curr_n_samples))
        padding = slab.Sound.silence(missing_length_ns, stim.samplerate, stim.n_channels)
        return slab.Sound.sequence(stim, padding)


# stim need to be standardized before feeding into cochleagram
# TODO: in the model, the cochleagram is generated at 48000 Hz
def normalize_binaural_stim(orig_stim: np.ndarray, orig_sr, target_sr=48000, scaling_max=0.1, min_len=2):
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
