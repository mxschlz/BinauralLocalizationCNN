"""
preprocessing for the Francl 2022 paper
things to do:
   normalize the sound
   pass the binaural sound through the Cochleagram
   make sure the shape of the data is (39, 48000, 2)
   down-sample the signals after the cochleagram

currently down sampling uses tensorflow conv2d
"""

import numpy as np
import scipy.signal as signallib
import tensorflow as tf

from pycochleagram import cochleagram as cgm
from stim_util import apply_hanning_window, normalize_binaural_stim

# default settings
NORM_SETTING = {'target_sr': 48000,
                'scaling_max': 0.1,
                'min_len': 2,
                }
COCH_SETTING = {'coch_gen_sig_cutoff': 2,
                'coch_freq_lims': (30, 20000),
                'minimum_padding': 0.35,
                'final_stim_length': 1,
                'hanning_windowed': True,
                'sliced': True,
                'channel_stack': True,
                }


# cochleagram
# NOTE: parameters signal_rate, coch_freq_lims, final_stim_length should NOT be changed to match the required model
#  inputs
# TODO: removing ILD/ITD should be before the cochleagram
def cochleagram_wrapper(stim, signal_rate=48000,
                        coch_gen_sig_cutoff=2, coch_freq_lims=(30, 20000),
                        minimum_padding=0.35, final_stim_length=1,
                        hanning_windowed=True, sliced=True, channel_stack=True):
    """
    pass the stimulus, stim through the cochleagram
    Args:
        stim: N-by-2 np array
        signal_rate: stimulus sampling rate
        coch_gen_sig_cutoff: maximum length of the stimulus to be used, second
        coch_freq_lims: 2 elements tuple/list/array, low and high limits of cochlea frequency
        minimum_padding: starting parts of the stimulus to be ignored, second
        final_stim_length: duration of the result, second
        hanning_windowed: if a hanning window is to be used to window the stimulus
        sliced: if pick a random `final_stim_length` duration portion of the stimulus
        channel_stack: if the resulting cochleagram should be stacked.
            if stacked, the result will be 36-N-2, otherwise 72-N

    Returns:
        resulting cochleagram, np array
    """
    # time to n samples conversion
    final_stim_length_n = round(final_stim_length * signal_rate)
    minimum_padding_n = round(minimum_padding * signal_rate)
    coch_gen_sig_cutoff_n = round(coch_gen_sig_cutoff * signal_rate)

    stim_wav = stim
    stim_freq = signal_rate

    # transpose to split channels to speed up calculating subbands
    stim_wav = stim_wav[:, :coch_gen_sig_cutoff_n]
    # delay = 15000
    # first dimesnion due to transpose
    total_singal = stim_wav.shape[1]
    # sample_factor is a Positive integer that determines how densely ERB function will be sampled to
    # create bandpass filters. see pycochleagram for more details
    sample_factor = 1

    # Apply a hanning window to the stimulus
    if hanning_windowed:
        hann_r = apply_hanning_window(stim_wav[1], 20, sample_rate=44100)
        hann_l = apply_hanning_window(stim_wav[0], 20, sample_rate=44100)
        r_channel = hann_r
        l_channel = hann_l
    else:
        r_channel = stim_wav[1]
        l_channel = stim_wav[0]

    # calculate subbands
    subbands_r = cgm.human_cochleagram(r_channel, stim_freq, low_lim=coch_freq_lims[0], hi_lim=coch_freq_lims[1],
                                       sample_factor=sample_factor, padding_size=10000,
                                       ret_mode='subband').astype(np.float32)
    subbands_l = cgm.human_cochleagram(l_channel, stim_freq, low_lim=coch_freq_lims[0], hi_lim=coch_freq_lims[1],
                                       sample_factor=sample_factor, padding_size=10000,
                                       ret_mode='subband').astype(np.float32)

    if sliced:
        front_limit = minimum_padding_n
        back_limit = total_singal - minimum_padding_n - final_stim_length_n
        jitter = np.random.randint(round(front_limit), round(back_limit))
        front_slice = jitter
        back_slice = jitter + final_stim_length_n
        # 44100*300ms = 13000
        subbands_l = subbands_l[:, front_slice:back_slice]
        subbands_r = subbands_r[:, front_slice:back_slice]

    if channel_stack:
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


def make_downsample_filt(old_sr=48000, new_sr=8000, wd_size=4097, beta=5.0,
                         pycoch_downsamp=False, tensor=False, double_channel=True):
    """
    Make the sinc filter that will be used to downsample the cochleagram
    :param old_sr: int, raw sampling rate of the audio signal
    :param new_sr: int, end sampling rate of the envelopes
    :param wd_size: int, the size of the downsampling window (should be large enough to go to zero on the edges)
    :param beta: float, kaiser window shape parameter
    :param pycoch_downsamp: Boolean, if uses pycochleagram downsampling function
    :param tensor: Boolean, if return a tensor
    :param double_channel: Boolean, if return a filter to filter 2 channels together. only works for tensor
    :return:
    """
    ds_factor = old_sr / new_sr
    if not pycoch_downsamp:
        downsample_filter_times = np.arange(-wd_size / 2, int(wd_size / 2))
        downsample_filter_response_orig = np.sinc(downsample_filter_times / ds_factor) / ds_factor
        downsample_filter_window = signallib.kaiser(wd_size, beta)
        downsample_filter_response = downsample_filter_window * downsample_filter_response_orig
    else:
        max_rate = ds_factor
        f_c = 1. / max_rate  # cutoff of FIR filter (rel. to Nyquist)
        half_len = 10 * max_rate  # reasonable cutoff for our sinc-like function
        if max_rate != 1:
            downsample_filter_response = signallib.firwin(2 * half_len + 1, f_c, window=('kaiser', beta))
        else:  # just in case we aren't downsampling -- I think this should work?
            downsample_filter_response = np.zeros(2 * half_len + 1)
            downsample_filter_response[half_len + 1] = 1

    # reshape the filter
    downsample_filter_response = downsample_filter_response.astype(np.float32)
    filt_len = downsample_filter_response.shape[0]
    if not double_channel:
        downsample_filter_response = downsample_filter_response.reshape(1, filt_len, 1, 1)
    else:
        x0 = np.zeros((1, filt_len, 1), dtype=np.float32)
        filt = downsample_filter_response.reshape(1, filt_len, 1)
        xl = np.concatenate([filt, x0], axis=2)
        xl = xl.reshape(1, filt_len, 2, 1)
        xr = np.concatenate([x0, filt], axis=2)
        xr = xr.reshape(1, filt_len, 2, 1)
        downsample_filter_response = np.concatenate([xl, xr], axis=3)

    if tensor:
        downsample_filter_response = tf.constant(downsample_filter_response, tf.float32)

    return downsample_filter_response


def downsample_tensor(signal, filter, ds_ratio, post_rectify=True):
    """
    use tensorflow convolution 2d to down-sample the cochleagram subbands
    :param signal: subbands returned from cochleagram_wrapper
    :param filter: tensor convolution filter
    :param ds_ratio: int, down-sample ratio
    :param post_rectify: bool, if rectifying after down-sample
    :return:
    """
    # use tensorflow convolution 2d
    # need to perform it separately on each channel
    # or, since we filter the 2 channels the same way, we can replicate the filter to 2 channels
    downsampled_signal = tf.nn.conv2d(signal, filter,
                                      strides=[1, 1, ds_ratio, 1], padding='SAME',
                                      name='conv2d_cochleagram_raw')
    if post_rectify:
        downsampled_signal = tf.nn.relu(downsampled_signal)

    return downsampled_signal


# not used, cannot get tensorflow pyfunc wrapper to work on cochleagram_wrapper
def process_single_stim(sig, ds_filter, ds_ratio):
    """
    use tensorflow conv2d to process single sound
    :param sig: binaural sound to be processed
    :return:
    """
    # run the stim through the cochleagram wrapper
    coch_subbands = cochleagram_wrapper(sig, )

    # check the shape; for tensorflow, expand the first dimension
    coch_subbands_reshaped = coch_subbands.reshape(1, *coch_subbands.shape)

    # down sample filtering
    return downsample_tensor(coch_subbands_reshaped, ds_filter, ds_ratio)


class data_generator():
    """
    create a generator to be used to construct the tf dataset
    :param data_dicts: list of dicts contains normalized sounds
    :return:
    """

    def __init__(self, data_dicts):
        self.data = data_dicts

    def __call__(self, *args, **kwargs):
        for d in self.data:
            yield d['subbands'], d['label']['hrtf_idx']


def process_stims(stim_dicts, norm_param={}, coch_param={}, filt_param={}, keep_ori=False):
    """
    perform preprocessing for the CNN from Francl et al 2022 paper
    :param stim_dicts: output from stim_gen
    :param norm_param: dict, see normalize_binaural_stim
    :param coch_param: dict, see cochleagram_wrapper
    :param filt_param: dict, see make_downsample_filt
    :param keep_ori: bool, if keep original sounds
    :return: the same stim_dicts, with cochleagram subbands added
    """

    if not isinstance(norm_param, dict) or \
            not isinstance(coch_param, dict) or \
            not isinstance(filt_param, dict):
        raise ValueError('parameters must be provided as dictionaries')

    target_sr = 48000
    if 'target_sr' in norm_param.keys():
        target_sr = int(norm_param['target_sr'])
        filt_param['old_sr'] = target_sr
        coch_param['signal_rate'] = target_sr

    # normalize each sounds
    for stim_d in stim_dicts:
        sr = stim_d['label']['sampling_rate']
        stim_d['sig'], _ = normalize_binaural_stim(stim_d['sig'], sr, **norm_param)

    new_sr = 8000
    if 'new_sr' in filt_param.keys():
        new_sr = int(filt_param['new_sr'])
        _ = filt_param.pop('new_sr')

    # run the cochleagram
    for stim_d in stim_dicts:
        coch_subbands = cochleagram_wrapper(stim_d['sig'], **coch_param)
        # check the shape; for tensorflow, expand the first dimension
        coch_subbands_reshaped = coch_subbands.reshape(1, *coch_subbands.shape)
        stim_d['subbands'] = coch_subbands_reshaped

        if not keep_ori:
            # remove original sound
            stim_d.pop('sig')

    # downsampling with tensorflow
    # it makes sense to perform everything in CPU here
    with tf.device('/cpu:0'):
        ds_ratio = int(target_sr / new_sr)
        ds_filter = make_downsample_filt(target_sr, new_sr, **filt_param, tensor=True)

        # construct a tf dataset from stim_dicts, and feed it into tensorflow down-sample
        data_gen = data_generator(stim_dicts)
        dataset = tf.data.Dataset.from_generator(data_gen,
                                                 output_types=(tf.float32, tf.int32),
                                                 output_shapes=(stim_dicts[0]['subbands'].shape, []))
        ds_iter = dataset.make_one_shot_iterator()
        subbands, hrtf_idx = ds_iter.get_next()
        # downsample
        subbands = downsample_tensor(subbands, ds_filter, ds_ratio)

    # start tensorflow session to do the work
    res = []
    with tf.Session() as sess:
        while True:
            try:
                res.append(sess.run([subbands, hrtf_idx]))
            except tf.errors.OutOfRangeError:
                break

    # now we can replace the subbands with downsampled copies
    for stim_d, downsampled in zip(stim_dicts, res):
        stim_d['subbands'] = downsampled[0]

    return stim_dicts


if __name__ == '__main__':
    pass
