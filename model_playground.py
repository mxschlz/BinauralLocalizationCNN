import logging
import shutil
import sys
from pathlib import Path

import coloredlogs
import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import visualkeras
from PIL import ImageFont
from keras import layers

from tf_explorer import ExplorerShell

logger = tf.get_logger()
logger.setLevel(logging.INFO)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

matplotlib.rcParams['figure.dpi'] = 360


# Example path: './resources/net_weights/net1/config_array.npy'
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


def text_callable(layer_index, layer):
    # Every other piece of text is drawn above the layer, the first one below
    above = bool(layer_index % 2)

    # Get the output shape of the layer
    output_shape = [x for x in list(layer.output_shape) if x is not None]

    # If the output shape is a list of tuples, we only take the first one
    if isinstance(output_shape[0], tuple):
        output_shape = list(output_shape[0])
        output_shape = [x for x in output_shape if x is not None]

    # Variable to store text which will be drawn
    output_shape_txt = ""

    layer_cfg = layer.get_config()
    if 'conv2d' in layer_cfg['name']:
        output_shape_txt += f"Conv2D"
        output_shape_txt += f"\nKernel size:\n{layer_cfg['kernel_size'][0]}x{layer_cfg['kernel_size'][1]}\nStrides:\n{layer_cfg['strides'][0]}x{layer_cfg['strides'][1]}"
    elif 'max_pooling2d' in layer_cfg['name']:
        output_shape_txt += f"Pool"
        output_shape_txt += f"\nPool size:\n{layer_cfg['pool_size'][0]}x{layer_cfg['pool_size'][0]}\nStrides:\n{layer_cfg['strides'][0]}x{layer_cfg['strides'][1]}"
    elif 'dense' in layer_cfg['name']:
        output_shape_txt += f"Dense"
        output_shape_txt += f"\n{layer_cfg['units']} units"
    elif 'dropout' in layer_cfg['name']:
        output_shape_txt += f"Dropout"
    elif 'batch_normalization' in layer_cfg['name']:
        output_shape_txt += f"BN"
    elif 'relu' in layer_cfg['name']:
        output_shape_txt += f"ReLU"

    # Create a string representation of the output shape
    output_shape_txt += "\nOutput shape:\n"
    for ii in range(len(output_shape)):
        output_shape_txt += str(output_shape[ii])
        if ii < len(output_shape) - 2:  # Add an x between dimensions, e.g. 3x3
            output_shape_txt += "x"
        if ii == len(output_shape) - 2:  # Add a newline between the last two dimensions, e.g. 3x3 \n 64
            output_shape_txt += "\n"

    # Return the text value and if it should be drawn above the layer
    return output_shape_txt, above


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

    model = keras.Sequential()
    # The input layer isn't really a layer, but a placeholder tensor w/ same shape as the input data
    # Network gets downsampled data (48kHz to 8kHz)
    model.add(keras.Input(shape=(39, 8000, 2), batch_size=16))

    for layer in config_array[0][1:]:
        logger.info(f'Adding layer: {layer}')
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
                logger.info('pad_along_height == 0')
                # Note: data_format='channels_last' is correctly inferred when adding the Conv2D layer
                model.add(layers.Conv2D(filters=filters, kernel_size=(kernel_height, kernel_width),
                                        strides=(stride_height, stride_width), padding='valid'))
            else:
                logger.info(f'pad_along_height != 0. pad_top, pad_bottom = {pad_top, pad_bottom}')
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
            # For training at least:
            # Shouldn't have softmax applied bc loss function is sparse_categorical_crossentropy_with_logits()
            # that needs logits, i.e., not normalized values and softmax produces normalized values

    model.load_weights(ckpt_path / 'model.ckpt-100000')
    return model


def show_diagram(model: keras.Sequential):
    img = visualkeras.layered_view(model, legend=True, text_callable=text_callable,
                                   font=ImageFont.load_default(size=36), scale_xy=0.5, min_xy=10, scale_z=0.5)
    plt.imshow(np.asarray(img))
    plt.show()


def migrate_checkpoints_to_TF2(path: Path, backup: bool = None):
    """
    Migrates TF1 checkpoints to TF2.
    Variables in TF1 checkpoints are stored and retrieved by name.
    Because TF2 doesn't have a (...) graph anymore, it needs a different naming scheme, containing
    full paths to the variables.

    Args:
        path: Path to the net_weights folder (e.g. Path('./resources/net_weights'))
        backup: Whether to create a backup of the net_weights folder before migrating, if None, asks the user
    """

    if backup is None:
        answer = input('Do you want to create a backup of the net_weights folder (~20GB)? (y/N)')
        backup = True if answer.lower() == 'y' else False
    if backup:
        shutil.copytree(path, path.with_suffix('.backup'))

    # Migrate all checkpoints
    for checkpoint in path.glob('net*'):
        _migrate_checkpoint(checkpoint / 'model.ckpt-100000')


def _migrate_checkpoint(path: Path):
    """
    Migrates a single TF1 checkpoint to TF2.
    Args:
        path: Path to the checkpoint file (e.g. ./resources/net_weights/net1/model.ckpt-100000)
    """
    logger.info(f'Migrating checkpoint: {path}')
    shell = ExplorerShell(path.as_posix())

    for child in sorted(shell._cwd.children, key=lambda x: x.name):
        name = child.name
        # Fully connected layer (Dense w/ 512 units)
        # All nets have the 512 units layer first and the 504 units layer second
        # wc_out_0/ should be dense/ and dense_1/
        if name == 'wc_fc_0':
            shell.do_mv('wc_fc_0 dense/kernel')
        elif name == 'wb_fc_0':
            shell.do_mv('wb_fc_0 dense/bias')
        # Output layer (Dense w/ 504 units)
        elif name == 'wc_out_0':
            shell.do_mv('wc_out_0 dense_1/kernel')
        elif name == 'wb_out_0':
            shell.do_mv('wb_out_0 dense_1/bias')
        # Conv2D kernel weights: rename e.g. wc_3/ to conv2d_3/kernel/
        elif 'wc_' in name:
            layer = name.split('_')[-1]
            if layer == '0':
                shell.do_mv(f'wc_0 conv2d/kernel')
            else:
                shell.do_mv(f'wc_{layer} conv2d_{layer}/kernel')
        # Conv2D biases: rename e.g. wb_layer/ to conv2d/kernel/
        elif 'wb_' in name:
            layer = name.split('_')[-1]
            if layer == '0':
                shell.do_mv(f'wb_0 conv2d/bias')
            else:
                shell.do_mv(f'wb_{layer} conv2d_{layer}/bias')

    shell.do_mutations('')  # List changes
    shell.do_commit('')  # Apply changes


if __name__ == '__main__':
    # Migrate checkpoints from TF1 to TF2
    # migrate_checkpoints_to_TF2(Path('resources/net_weights'))
    # sys.exit()

    # Create model
    model = create_model(Path('./resources/net_weights/net1'))
    show_diagram(model)

    # Create models
    # models = create_models(Path('./resources/net_weights'))
    # for model in models:
    #     show_diagram(models[model])
