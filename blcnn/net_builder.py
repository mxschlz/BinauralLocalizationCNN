import collections
import json
from pathlib import Path
import logging

import keras
import numpy as np
import tensorflow as tf
from google.protobuf.json_format import MessageToJson
from keras import layers
from scipy import signal as signallib
import coloredlogs

logger = tf.get_logger()
logger.setLevel(logging.DEBUG)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def create_models(net_weights_path: Path) -> dict:
    """
    Load all config arrays from the given path to weights
    Args:
        net_weights_path: Path to the net_weights/ directory downloaded from DropBox (e.g. ./resources/net_weights)
                          (see https://github.com/afrancl/BinauralLocalizationCNN)
    Returns:
        dict: Dictionary with net IDs (1-10) as keys and keras.Sequential models as values

    """
    models = dict()
    for net_id in range(1, 11):
        ckpt_path = net_weights_path / f'net{net_id}'
        models[net_id] = create_model(ckpt_path)
    return models


def create_model(ckpt_path: Path) -> keras.Sequential:
    """
    Iterates over the config array in the given directory and creates a model from it.
    Populates the model with the weights from the checkpoint file and returns it.

    Model reverse engineered in TF2 from TF1 checkpoint files.
    Convolutional layers have custom padding to match the original network.
    Args:
        ckpt_path: Path to the checkpoint directory with the config_array.npy and model.ckpt-100000 (data and index,
                   meta not required) files for one network (e.g. ./resources/net_weights/net1)
    Returns:
        keras.Sequential: Model created from the config array and weights
    """

    logger.info(f'Building network from path: {ckpt_path}')

    config_array = np.load(ckpt_path / 'config_array.npy', allow_pickle=True)

    model = tf.keras.Sequential()
    # The input layer isn't really a layer, but a placeholder tensor w/ same shape as the input data
    # Network gets downsampled data (48kHz to 8kHz)
    model.add(keras.Input(shape=(39, 8000, 2), batch_size=16, name='train/image'))

    for layer in config_array[0][1:]:
        logger.debug(f'Adding layer: {layer}')
        if layer[0] == 'conv':
            # Input shape for Conv2D must be: (batch_size, imageside1, imageside2, channels)
            # config_array: ['conv', [2, 32, 32], [1, 1]] -> [Conv2D, [kernel_height, kernel_width, filters], [stride_height, stride_width]]

            # Get height of the previous layer (39 for first layer)
            pl_height = model.layers[-1].get_output_shape_at(0)[1] if model.layers else 39

            # filters is the dimensionality of the output space (i.e. the number of output filters in the convolution)

            kernel_height, kernel_width, filters = layer[1]
            stride_height, stride_width = layer[2]

            # Custom padding
            if (pl_height % stride_height == 0):
                pad_along_height = max(kernel_height - stride_height, 0)
            else:
                pad_along_height = max(kernel_height - (pl_height % stride_height), 0)
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top

            if pad_along_height == 0:  # or padding == 'SAME': # -> SAME padding is not used for now
                # logger.debug('pad_along_height == 0')
                # Note: data_format='channels_last' is correctly inferred when adding the Conv2D layer
                model.add(layers.Conv2D(filters=filters, kernel_size=(kernel_height, kernel_width),
                                        strides=(stride_height, stride_width), padding='valid'))
            else:
                # logger.debug(f'pad_along_height != 0. pad_top, pad_bottom = {pad_top, pad_bottom}')
                model.add(layers.ZeroPadding2D(padding=((pad_top, pad_bottom), (0, 0))))
                model.add(layers.Conv2D(filters=filters, kernel_size=(kernel_height, kernel_width),
                                        strides=(stride_height, stride_width), padding='valid'))
        elif layer[0] == 'relu' or layer[0] == 'fc_relu':
            # config_array: ['relu'] or ['fc_relu'] -> [ReLU]
            model.add(layers.ReLU())
        elif layer[0] == 'bn':
            # config_array: ['bn'] -> [BatchNormalization]
            model.add(layers.BatchNormalization())
        elif layer[0] == 'pool':
            # config_array: ['pool', [1, 8]] -> [layer_name, [kernel_height, kernel_width]]
            # Non-overlapping kernel strides -> strides = kernel_size
            kernel_size = layer[1]
            model.add(layers.MaxPool2D(pool_size=kernel_size, strides=kernel_size, padding='valid'))
        elif layer[0] == 'fc':
            # config_array: ['fc', 512] -> [FullyConnected, units]
            # Input shape must be: (batch_size, input_size) -> flatten the output of the previous layer first
            # (This is a huge layer, comprises about ~90% of the model's parameters by the looks of it)
            model.add(layers.Flatten())
            units = layer[1]
            model.add(layers.Dense(units=units))
        elif layer[0] == 'fc_bn':
            # config_array: ['fc_bn'] -> [BatchNormalization]
            # Original code casts it to filter_dtype=tf.float32 after (in comp. to 'bn')
            # Look at .build() call to see what's passed as filter_dtype
            model.add(layers.BatchNormalization())
        elif layer[0] == 'dropout':
            # config_array: ['dropout'] -> [Dropout]
            model.add(layers.Dropout(rate=0.5))
        elif layer[0] == 'out':
            # config_array: ['out'] -> [FullyConnected, 504 units]
            model.add(layers.Dense(units=504,
                                   activation='softmax'))
            # Pick unit with highest probability
            # model.add(layers.Lambda(lambda x: tf.argmax(x, axis=1)))  # Experimental!

    # TODO: Check if optimizers need to be ported to TF2
    # model.compile(optimizer=keras.optimizers.legacy.Adam(),  # Optimizer
    #               # Loss function to minimize
    #               loss=keras.losses.SparseCategoricalCrossentropy(),  # List of metrics to monitor
    #               metrics=[keras.metrics.SparseCategoricalAccuracy()], )
    return model


def get_feature_dict(tf_file, is_bkgd=False):
    """
    get the feature dict needed to construct dataset iterator from tfrecords
    :param tf_file: a single sample tfrecords file
    :param tf_opts: compression used in the tfrecords file
    :param is_bkgd: bool, if the tfrecords is about background sound
    :return:
    """
    sample_record = tf.data.TFRecordDataset(tf_file, compression_type='GZIP').take(1).get_single_element().numpy()
    samp_js = MessageToJson(tf.train.Example.FromString(sample_record))
    jsdict = json.loads(samp_js)

    feature = collections.OrderedDict()
    for _, v in sorted(jsdict.items()):
        v = v['feature']
        key1 = v.keys()
        for x in key1:
            key2 = v[x].keys()
            for y in key2:
                if y == 'int64List':
                    val_dtype = tf.int64
                elif y == 'bytesList':
                    val_dtype = tf.string
                elif y == 'floatList':
                    val_dtype = tf.float32
                else:
                    raise KeyError("conversion of data type {} not implemented".format(y))
                feature_len = len(v[x][y]['value'])
                shape = [] if feature_len == 1 else [feature_len]
                if is_bkgd is True and (x == 'train/azim' or x == 'train/elev'):
                    feature[x] = tf.io.VarLenFeature(val_dtype)
                else:
                    feature[x] = tf.io.FixedLenFeature(shape, val_dtype)

    return feature


def single_example_parser(example):
    feature_description = {
        'train/azim': tf.io.FixedLenFeature([], tf.int64),
        'train/elev': tf.io.FixedLenFeature([], tf.int64),
        'train/image': tf.io.FixedLenFeature([], tf.string),
        'train/image_height': tf.io.FixedLenFeature([], tf.int64),
        'train/image_width': tf.io.FixedLenFeature([], tf.int64),
        'train/name': tf.io.FixedLenFeature([], tf.string)}
    example = tf.io.parse_single_example(example, feature_description)
    example['train/image'] = tf.reshape(tf.io.decode_raw(example['train/image'], tf.float32), (39, 48000, 2))
    example['train/name'] = tf.io.decode_compressed(example['train/name'])

    [L_channel, R_channel] = tf.unstack(example['train/image'], axis=2)
    concat_for_downsample = tf.concat([L_channel, R_channel], axis=0)
    reshaped_for_downsample = tf.expand_dims(concat_for_downsample, axis=2)

    # hard coding filter shape based on previous experimentation
    new_sig_downsampled = downsample(reshaped_for_downsample, 48000, 8000, window_size=4097, beta=10.06,
                                     post_rectify=True)
    # ###
    # downsample = 6
    #
    # # TODO: See if the downsample tensor needs to be recreated for every sample
    # # downsample_filter_tensor = make_downsample_filt_tensor(current_rate=48000, new_rate=8000,
    # #                                                        window_size=4097, beta=10.06)
    # downsample_filter_times = np.arange(-4097 / 2, int(4097 / 2))
    # downsample_filter_response_orig = np.sinc(downsample_filter_times / downsample) / downsample
    # downsample_filter_window = signallib.windows.kaiser(4097, 10.06)
    # downsample_filter_response = tf.convert_to_tensor(downsample_filter_window * downsample_filter_response_orig, tf.float32)
    # # downsample_filt_tensor = tf.constant(downsample_filter_response, tf.float32)
    # downsample_filt_tensor = tf.expand_dims(downsample_filter_response, 0)
    # downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 2)
    # downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 3)
    #
    # signal = tf.expand_dims(reshaped_for_downsample, 0)
    # new_sig_downsampled = tf.nn.conv2d(signal, downsample_filt_tensor, strides=[1, 1, downsample, 1], padding='SAME',
    #                                   name='conv2d_cochleagram_raw')
    #
    #
    #
    # ###
    #

    downsampled_squeezed = tf.squeeze(new_sig_downsampled)
    [L_channel_downsampled, R_channel_downsampled] = tf.split(downsampled_squeezed, num_or_size_splits=2, axis=0)
    downsampled_reshaped = tf.stack([L_channel_downsampled, R_channel_downsampled], axis=2)
    example['train/image'] = tf.pow(downsampled_reshaped, 0.3)

    example['target'] = (
        tf.add(
            tf.multiply(
                tf.constant(72, dtype=tf.int64),
                tf.math.floordiv(
                    example['train/elev'],
                    tf.constant(10, dtype=tf.int64)
                )
            ),
            tf.math.floordiv(
                example['train/azim'],
                tf.constant(5, dtype=tf.int64)
            )
        )
    )

    return example['train/image'], example['target'], example['train/name'] #, example['train/azim'], example['train/elev']


def make_downsample_filt_tensor(current_rate=48000, new_rate=8000, window_size=4097, beta=10.06):
    """
    Make the sinc filter that will be used to downsample the cochleagram
    Parameters
    ----------
    current_rate : int
        raw sampling rate of the audio signal
    new_rate : int
        end sampling rate of the envelopes
    window_size : int
        the size of the downsampling window (should be large enough to go to zero on the edges).
    beta : float
        kaiser window shape parameter
    Returns
    -------
    downsample_filt_tensor : tensorflow tensor, tf.float32
        a tensor of shape [0 WINDOW_SIZE 0 0] the sinc windows with a kaiser lowpass filter that is applied while downsampling the cochleagram
    """
    ds_ratio = current_rate / new_rate
    downsample_filter_times = np.arange(-window_size / 2, int(window_size / 2))
    downsample_filter_response_orig = np.sinc(downsample_filter_times / ds_ratio) / ds_ratio
    downsample_filter_window = signallib.windows.kaiser(window_size, beta)
    downsample_filter_response = downsample_filter_window * downsample_filter_response_orig
    downsample_filt_tensor = tf.constant(downsample_filter_response, tf.float32)
    downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 0)
    downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 2)
    downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 3)

    return downsample_filt_tensor


def downsample(signal, current_rate, new_rate, window_size, beta, post_rectify=True):
    downsample = current_rate / new_rate
    message = ("The current downsample rate {} is "
               "not an integer. Only integer ratios "
               "between current and new sampling rates "
               "are supported".format(downsample))

    assert (current_rate % new_rate == 0), message
    message = ("New rate must be less than old rate for this "
               "implementation to work!")
    assert (new_rate < current_rate), message
    # TODO: See if the downsample tensor needs to be recreated for every sample
    downsample_filter_tensor = make_downsample_filt_tensor(current_rate=current_rate, new_rate=new_rate,
                                                           window_size=window_size, beta=beta)
    signal = tf.expand_dims(signal, 0)
    downsampled_signal = tf.nn.conv2d(signal, downsample_filter_tensor, strides=[1, 1, downsample, 1], padding='SAME',
                                      name='conv2d_cochleagram_raw')
    # downsampled_signal = keras.layers.Conv2D(filters=1, kernel_size=(window_size, 1),
    #                                          strides=(1, int(downsample)),
    #                                          padding='same')(signal)
    if post_rectify:
        downsampled_signal = tf.nn.relu(downsampled_signal)

    return downsampled_signal
