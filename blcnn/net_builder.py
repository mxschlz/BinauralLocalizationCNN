import collections
import json
from pathlib import Path
import logging

import keras
import numpy as np
import tensorflow as tf
from google.protobuf.json_format import MessageToJson
from keras import layers, regularizers
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
    # for net_id in range(1, 11):
    for net_id in range(1, 2):
        ckpt_path = net_weights_path / f'net{net_id}'
        models[net_id] = create_checkpointed_model(ckpt_path)
    return models


def create_checkpointed_model_functional(ckpt_path: Path) -> keras.Model:
    """
    CHECKPOINTED VERSION
    Iterates over the config array in the given directory and creates a model from it.
    Populates the model with the weights from the checkpoint file and returns it.

    Model reverse engineered in TF2 from TF1 checkpoint files.
    Convolutional layers have custom padding to match the original network.
    Args:
        ckpt_path: Path to the checkpoint directory with the config_array.npy and model.ckpt-100000 (data and index,
                   meta not required) files for one network (e.g. ./resources/net_weights/net1)
    Returns:
        keras.Model: Functional model created from the config array and weights
    """

    logger.info(f'Building network from path: {ckpt_path}')

    config_array = np.load(ckpt_path / 'config_array.npy', allow_pickle=True)

    inputs = keras.Input(shape=(39, 8000, 2), batch_size=16, name='image')
    x = inputs

    current_block = []  # Holds lambdas that build layers

    def flush_block():
        nonlocal current_block
        nonlocal x
        if current_block:
            x = make_checkpointed_block(current_block)(x)
            current_block = []
            return x
        return None

    for layer in config_array[0][1:]:
        kind = layer[0]

        logger.debug(f'Processing layer: {layer}')
        # logger.debug(f'Model layers: {[l.name for l in x.layers]}')

        if kind == 'conv':
            kernel_height, kernel_width, filters = layer[1]
            stride_height, stride_width = layer[2]

            # Custom padding computation
            pl_height = x.shape[1] if len(x.shape) > 1 else 39
            if (pl_height % stride_height == 0):
                pad_along_height = max(kernel_height - stride_height, 0)
            else:
                pad_along_height = max(kernel_height - (pl_height % stride_height), 0)
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top

            if pad_along_height == 0:
                current_block.append(lambda kh=kernel_height, kw=kernel_width, f=filters, sh=stride_height, sw=stride_width:
                                     layers.Conv2D(filters=f, kernel_size=(kh, kw), strides=(sh, sw), padding='valid'))
            else:
                current_block.append(lambda pt=pad_top, pb=pad_bottom:
                                     layers.ZeroPadding2D(padding=((pt, pb), (0, 0))))
                current_block.append(lambda kh=kernel_height, kw=kernel_width, f=filters, sh=stride_height, sw=stride_width:
                                     layers.Conv2D(filters=f, kernel_size=(kh, kw), strides=(sh, sw), padding='valid'))
        elif kind in ('relu', 'fc_relu'):
            current_block.append(lambda: layers.ReLU())
        elif kind == 'bn':
            current_block.append(lambda: layers.BatchNormalization())
        elif kind == 'fc':
            current_block.append(lambda: layers.Flatten())
            units = layer[1]
            current_block.append(lambda u=units: layers.Dense(units=u))
        elif kind == 'fc_bn':
            current_block.append(lambda: layers.BatchNormalization())
        elif kind == 'dropout':
            current_block.append(lambda: layers.Dropout(rate=0.5))
        elif kind == 'out':
            units = 504
            current_block.append(lambda u=units: layers.Dense(units=u, activation='softmax'))
        elif kind == 'pool':
            # 🧠 Flush current block as checkpointed before adding pooling
            x = flush_block()
            kernel_size = layer[1]
            x = layers.MaxPool2D(pool_size=kernel_size, strides=kernel_size, padding='valid')(x)
    # Final flush if any layers are left
    x = flush_block()
    # Create the model
    model = keras.Model(inputs=inputs, outputs=x)
    # Print out the model summary to check if it was built correctly
    model.build()
    logger.info(model.summary())
    return model


def create_checkpointed_model(ckpt_path: Path) -> keras.Sequential:
    """
    CHECKPOINTED VERSION
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
    print("Eager execution enabled:", tf.executing_eagerly())
    model = tf.keras.Sequential()
    print("Eager execution enabled:", tf.executing_eagerly())
    model.add(keras.Input(shape=(39, 8000, 2), batch_size=16, name='image'))
    print("Eager execution enabled:", tf.executing_eagerly())

    current_block = []  # Holds lambdas that build layers

    def flush_block():
        nonlocal current_block
        if current_block:
            print("Eager execution enabled:", tf.executing_eagerly())
            model.add(make_checkpointed_block(current_block))
            current_block = []

    for layer in config_array[0][1:]:
        kind = layer[0]

        logger.debug(f'Processing layer: {layer}')
        logger.debug(f'Model layers: {[l.name for l in model.layers]}')

        if kind == 'conv':
            kernel_height, kernel_width, filters = layer[1]
            stride_height, stride_width = layer[2]

            # Custom padding computation
            pl_height = model.layers[-1].get_output_shape_at(0)[1] if model.layers else 39
            if (pl_height % stride_height == 0):
                pad_along_height = max(kernel_height - stride_height, 0)
            else:
                pad_along_height = max(kernel_height - (pl_height % stride_height), 0)
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top

            if pad_along_height == 0:
                current_block.append(lambda kh=kernel_height, kw=kernel_width, f=filters, sh=stride_height, sw=stride_width:
                                     layers.Conv2D(filters=f, kernel_size=(kh, kw), strides=(sh, sw), padding='valid'))
            else:
                current_block.append(lambda pt=pad_top, pb=pad_bottom:
                                     layers.ZeroPadding2D(padding=((pt, pb), (0, 0))))
                current_block.append(lambda kh=kernel_height, kw=kernel_width, f=filters, sh=stride_height, sw=stride_width:
                                     layers.Conv2D(filters=f, kernel_size=(kh, kw), strides=(sh, sw), padding='valid'))

        elif kind in ('relu', 'fc_relu'):
            current_block.append(lambda: layers.ReLU())
        elif kind == 'bn':
            current_block.append(lambda: layers.BatchNormalization())
        elif kind == 'fc':
            current_block.append(lambda: layers.Flatten())
            units = layer[1]
            current_block.append(lambda u=units: layers.Dense(units=u))
        elif kind == 'fc_bn':
            current_block.append(lambda: layers.BatchNormalization())
        elif kind == 'dropout':
            current_block.append(lambda: layers.Dropout(rate=0.5))
        elif kind == 'out':
            units = 504
            current_block.append(lambda u=units: layers.Dense(units=u, activation='softmax'))

        elif kind == 'pool':
            # 🧠 Flush current block as checkpointed before adding pooling

            print("Eager execution enabled:", tf.executing_eagerly())
            flush_block()
            kernel_size = layer[1]
            model.add(layers.MaxPool2D(pool_size=kernel_size, strides=kernel_size, padding='valid'))

    # Final flush if any layers are left
    flush_block()

    # Print out the model summary to check if it was built correctly
    model.build()
    logger.info(model.summary())

    return model


def make_checkpointed_block(layer_fns):
    class Block(tf.keras.layers.Layer):
        def __init__(self):
            print("Eager execution enabled:", tf.executing_eagerly())
            super().__init__()
            self.layers = [fn() for fn in layer_fns]

        @tf.recompute_grad
        # @tf.function
        def call(self, x):
            print("Eager execution enabled test:", tf.executing_eagerly())
            for layer in self.layers:
                x = layer(x)
            return x
    return Block()


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
    model.add(keras.Input(shape=(39, 8000, 2), batch_size=16, name='train/image'))  # TF2.14
    # model.add(keras.Input(shape=(39, 8000, 2), batch_size=16, name='image'))  # TF2.16

    reg = regularizers.L1(l1=0.001)
    # reg = None

    for layer in config_array[0][1:]:
        logger.debug(f'Adding layer: {layer}')
        if layer[0] == 'conv':
            # Input shape for Conv2D must be: (batch_size, imageside1, imageside2, channels)
            # config_array: ['conv', [2, 32, 32], [1, 1]] -> [Conv2D, [kernel_height, kernel_width, filters], [stride_height, stride_width]]

            # Get height of the previous layer (39 for first layer)
            pl_height = model.layers[-1].get_output_shape_at(0)[1] if model.layers else 39  # TF2.14
            # pl_height = model.layers[-1].output.shape[1] if model.layers else 39  # TF2.16

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
                                        strides=(stride_height, stride_width), padding='valid',
                                        kernel_regularizer=reg))
            else:
                # logger.debug(f'pad_along_height != 0. pad_top, pad_bottom = {pad_top, pad_bottom}')
                model.add(layers.ZeroPadding2D(padding=((pad_top, pad_bottom), (0, 0))))
                model.add(layers.Conv2D(filters=filters, kernel_size=(kernel_height, kernel_width),
                                        strides=(stride_height, stride_width), padding='valid',
                                        kernel_regularizer=reg))
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
            model.add(layers.Dense(units=units, kernel_regularizer=reg))
        elif layer[0] == 'fc_bn':
            # config_array: ['fc_bn'] -> [BatchNormalization]
            # Original code casts it to filter_dtype=tf.float32 after (in comp. to 'bn')
            # Look at .build() call to see what's passed as filter_dtype
            model.add(layers.BatchNormalization(momentum=0.9))
        elif layer[0] == 'dropout':
            # config_array: ['dropout'] -> [Dropout]
            model.add(layers.Dropout(rate=0.5))
        elif layer[0] == 'out':
            # config_array: ['out'] -> [FullyConnected, 504 units]
            model.add(layers.Dense(units=504,
                                   activation='softmax',
                                   kernel_regularizer=reg))
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
                if is_bkgd is True and (x == 'azim' or x == 'elev'):
                    feature[x] = tf.io.VarLenFeature(val_dtype)
                else:
                    feature[x] = tf.io.FixedLenFeature(shape, val_dtype)

    return feature


def single_example_parser(example):
    feature_description = {
        # 'azim': tf.io.FixedLenFeature([], tf.int64),
        # 'elev': tf.io.FixedLenFeature([], tf.int64),

        'train/image': tf.io.FixedLenFeature([], tf.string),  # TF2.14
        # 'image': tf.io.FixedLenFeature([], tf.string),  # TF2.16

        # 'image_height': tf.io.FixedLenFeature([], tf.int64),
        # 'image_width': tf.io.FixedLenFeature([], tf.int64),
        # 'name': tf.io.FixedLenFeature([], tf.string)

        'train/target': tf.io.FixedLenFeature([], tf.int64)  #TF2.14
        # 'target': tf.io.FixedLenFeature([], tf.int64)  #TF2.16
        }
    example = tf.io.parse_single_example(example, feature_description)
    example['train/image'] = tf.reshape(tf.io.decode_raw(example['train/image'], tf.float32), (39, 8000, 2))  #TF2.14
    # example['image'] = tf.reshape(tf.io.decode_raw(example['image'], tf.float32), (39, 8000, 2))  #TF2.16
    # example['name'] = tf.io.decode_compressed(example['name'])

    return example['train/image'], example['train/target']#, example['name'] #, example['azim'], example['elev']  #TF2.14
    # return example['image'], example['target']#, example['name'] #, example['azim'], example['elev']  #TF2.16


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


