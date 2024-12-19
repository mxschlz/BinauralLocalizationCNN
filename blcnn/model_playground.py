import csv
import logging
import shutil
import time
from pathlib import Path

import coloredlogs
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import visualkeras
from PIL import ImageFont

from net_builder import create_model, get_feature_dict, single_example_parser
from tf_explorer import ExplorerShell

logger = tf.get_logger()
logger.setLevel(logging.DEBUG)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main() -> None:
    # migrate_checkpoints_to_TF2(Path('resources/Binaural Localization Net Weights'))
    # _migrate_checkpoint(Path('resources/Binaural Localization Net Weights/net2/model.ckpt-100000'))
    # sys.exit()
    # Disable GPU
    logger.info(f'Physical devices: {tf.config.list_physical_devices()}')
    only_cpu = False
    if only_cpu:
        tf.config.set_visible_devices([], 'GPU')
        logger.info('Removing GPU devices to force CPU usage')
    logger.info(f'Visible devices: {tf.config.get_visible_devices()}')

    # print(data)
    # plot_accuracy_grid(data, style='debug')
    # sys.exit()




    # train_dataset = tf.data.Dataset.from_tensor_slices(([0,1,0,0,1], [1,0,0,0,1]))
    # for row in train_dataset.take(1):
    #     print(row)
    # sys.exit()


    # Migrate checkpoints from TF1 to TF2
    # migrate_checkpoints_to_TF2(Path('resources/net_weights'))
    # sys.exit()

    # Create model
    model = create_model(Path('models/tf1/net_weights/net1'))
    # show_diagram(model)

    # Create models
    # models = create_models(Path('./resources/net_weights'))
    # for model in models:
    #     show_diagram(models[model])

    # Important information
    # which hrtf was used & when was that brir set created (i.e. which one is it if there's more than 1)
    # which net was used
    # when was the inference started

    # Load dataset
    feature_dict = get_feature_dict(
        'data/processed/cochleagrams_2024-12-17_19-14-28/cochs_hrtf_slab_default_kemar.tfrecord')
    logger.info(f'Features in supplied tfrecord file: {feature_dict}')

    # TODO: Better performance by batching Example protos and using parse_example; see if useful
    dataset_name = 'cochleagrams_2024-12-18_02-36-53/cochs_hrtf_b_nh2.tfrecord'
    # dataset = (tf.data.TFRecordDataset('data/_wrong_coords/training_data_2024-09-23_16-38-47/training-data_hrtf-default.tfrecord', compression_type="GZIP")
    dataset = (tf.data.TFRecordDataset(f'data/{dataset_name}', compression_type="GZIP").map(
        lambda serialized_example: single_example_parser(serialized_example)).shuffle(64).batch(16, drop_remainder=True).prefetch(1))

    pred_classes, true_classes = model.predict(dataset)
    print('Pred classes shape:', pred_classes.shape)
    print('True classes shape:', true_classes.shape)
    print('Pred classes:', pred_classes)
    print('True classes:', true_classes)


    # Argmax to get the predicted class
    pred_classes = np.argmax(pred_classes, axis=1)

    print('pred_classes = [', *pred_classes, ']', sep=', ')
    print('true_classes =', *true_classes)
    # write to CSV
    Path('data/output').mkdir(exist_ok=True)
    Path(f'data/output/for_cochleagrams_2024-12-18_02-36-53').mkdir(exist_ok=True)
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    with open(f'output/for_cochleagrams_2024-12-18_02-36-53/net1_{timestamp}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true_class', 'pred_class'])
        writer.writerows(zip(true_classes, pred_classes))


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
    main()

