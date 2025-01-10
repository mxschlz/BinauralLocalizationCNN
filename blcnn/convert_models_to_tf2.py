import logging
from pathlib import Path

import coloredlogs
import tensorflow as tf

from net_builder import create_model

logger = tf.get_logger()
logger.setLevel(logging.DEBUG)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main() -> None:
    convert_tf1_checkpoints_to_tf2_keras_models(Path('models/Binaural Localization Net Weights'))


def convert_tf1_checkpoints_to_tf2_keras_models(path: Path) -> None:
    """Convert TF1 checkpoints downloaded from Dropbox to Keras models in TF2.
    Args:
        path: Path to the directory containing the checkpoints for the 10 models.
    """
    for i in range(1, 11):
        checkpoint_path = path / f'net{i}/model.ckpt-100000'
        logger.info(f'Converting checkpoint {checkpoint_path} to Keras model...')
        _convert_ckpt_to_model(checkpoint_path, f'models/keras/net{i}.keras')
        logger.info(f'Converted checkpoint for net{i} to Keras model at models/keras/net{i}.keras\n')


def _convert_ckpt_to_model(checkpoint_path: Path, output_path: str) -> None:
    """Converts a TF1 checkpoint to a Keras model in TF2.
    Based on: https://www.tensorflow.org/guide/migrate/migrating_checkpoints#convert_tf1_checkpoint_to_tf2

    Args:
      checkpoint_path: Path to the TF1 checkpoint.
      output_path: Path to save the converted Keras model.
    """
    reader = tf.train.load_checkpoint(checkpoint_path)
    tf1_variables = tf.train.list_variables(checkpoint_path)

    tf2_model = create_model(Path(checkpoint_path).parent)

    logger.debug(f'TF1 checkpoint variables: {tf1_variables}')
    logger.debug(f'TF2 model trainable variables: {[v.name for v in tf2_model.trainable_variables]}')
    logger.debug(f'TF2 model non-trainable variables: {[v.name for v in tf2_model.non_trainable_variables]}')

    for tf1_name, shape in tf1_variables:
        # Skip optimizer variables bc their format apparently isn't compatible with TF2
        if "/Adam" in tf1_name or tf1_name in {"beta1_power", "beta2_power"}:
            continue

        # Get the TF2 variable name
        tf2_name = _replace_name(tf1_name)

        # Find the corresponding variable in the TF2 model
        tf2_variable = None
        for var in tf2_model.trainable_variables + tf2_model.non_trainable_variables:
            if var.name.split(":")[0] == tf2_name:
                tf2_variable = var
                break

        if tf2_variable is None:
            logger.warning(f"Warning: No matching TF2 variable found for {tf1_name} -> {tf2_name}")
            continue

        # Load the value from the TF1 checkpoint and assign it
        value = reader.get_tensor(tf1_name)
        tf2_variable.assign(value)
        logger.debug(f"Mapped {tf1_name} -> {tf2_name}")

    # Save as Keras model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tf2_model.save(output_path)

    # Clear the session to reset numbering of layers in the TF2 model
    tf.keras.backend.clear_session()


def _replace_name(name: str) -> str:
    """Replaces the names of the variables in the TF1 checkpoint with the names of the variables in the TF2 model.

    Args:
        name: Name of the variable in the TF1 checkpoint.

    Returns:
        Name of the variable in the TF2 model.
    """
    if 'wc_fc_0' in name:
        return name.replace('wc_fc_0', 'dense/kernel')
    elif 'wb_fc_0' in name:
        return name.replace('wb_fc_0', 'dense/bias')
    elif 'wc_out_0' in name:
        return name.replace('wc_out_0', 'dense_1/kernel')
    elif 'wb_out_0' in name:
        return name.replace('wb_out_0', 'dense_1/bias')
    elif 'wc_' in name:
        layer = name.split('_')[-1]
        if layer == '0':
            return name.replace(f'wc_0', 'conv2d/kernel')
        else:
            return name.replace(f'wc_{layer}', f'conv2d_{layer}/kernel')
    elif 'wb_' in name:
        layer = name.split('_')[-1]
        if layer == '0':
            return name.replace(f'wb_0', 'conv2d/bias')
        else:
            return name.replace(f'wb_{layer}', f'conv2d_{layer}/bias')
    else:
        return name


if __name__ == "__main__":
    main()
