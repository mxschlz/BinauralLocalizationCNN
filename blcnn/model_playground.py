import csv
import datetime
import glob
import logging
import pprint
import time
from pathlib import Path
from typing import List

import coloredlogs
import keras
import numpy as np
import tensorflow as tf

from util import get_unique_folder_name, load_config, ModelPlaygroundConfig
from net_builder import single_example_parser

logger = tf.get_logger()
logger.setLevel(logging.INFO)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main() -> None:
    """
    Main entry point for the model playground script.
    Loads config, disables GPU if needed, and runs the testing for each HRTF label.
    """
    # Disable GPU if needed
    logger.info(f'Physical devices: {tf.config.list_physical_devices()}')
    only_cpu = False
    if only_cpu:
        tf.config.set_visible_devices([], 'GPU')
        logger.info('Removing GPU devices to force CPU usage')
    logger.info(f'Visible devices: {tf.config.get_visible_devices()}')

    # Load config
    eval_config = load_config('blcnn/config.yml').model_playground
    logger.info(f'Loaded config: {eval_config}')

    for hrtf_label in eval_config.hrtf_labels:
        logger.info(f'Testing with HRTF: {hrtf_label}')
        test_multiple_models(hrtf_label, eval_config)


def test_multiple_models(hrtf_label: str, eval_config: ModelPlaygroundConfig) -> None:
    """
    Run the testing for one HRTF label using the models specified in the config.
    """
    start_time = time.time()
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')

    path_to_models = Path('models/keras/')
    path_to_cochleagrams = Path(f'data/cochleagrams/{hrtf_label}')

    dest = get_unique_folder_name(f'data/output/{path_to_cochleagrams.name}/')
    Path(dest).mkdir(parents=True, exist_ok=False)

    for model_id in eval_config.models_to_use:
        model_path = path_to_models / f'net{model_id}.keras'
        test_single_model(model_path, path_to_cochleagrams, dest)

    elapsed_time = str(datetime.timedelta(seconds=time.time() - start_time))
    summary = summarize_testing(eval_config, path_to_cochleagrams, timestamp, elapsed_time)
    logger.info(summary)
    with open(dest / f'_summary_{timestamp}.txt', 'w') as f:
        f.write(summary)


def test_single_model(model_path: Path = None, path_to_cochleagrams: Path = None, dest: Path = None):
    """
    Test a single model with the given cochleagrams and save the results to a CSV file.
    """
    logger.info(f'Testing model: {model_path}, with cochleagrams from: {path_to_cochleagrams}, saving to: {dest}')

    model = keras.models.load_model(model_path)

    # TODO: Better performance by batching Example protos and using parse_example; see if useful
    dataset = (
        tf.data.TFRecordDataset(path_to_cochleagrams / 'cochleagrams.tfrecord', compression_type="GZIP")
        .map(lambda serialized_example: single_example_parser(serialized_example))
        # .shuffle(64)
        .batch(16, drop_remainder=True)
        .prefetch(1)
    )

    # Predict
    true_classes = []
    pred_classes = []

    for predictions, labels in predict_with_ground_truth(model, dataset):
        true_classes.append(labels.numpy())
        pred_classes.append(predictions.numpy())

    true_classes = np.concatenate(true_classes, axis=0)
    pred_classes = np.concatenate(pred_classes, axis=0).argmax(axis=1)

    # write to CSV
    with open(dest / f'{model_path.name.split(".")[0]}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true_class', 'pred_class'])
        writer.writerows(zip(true_classes, pred_classes))


def summarize_testing(eval_config: ModelPlaygroundConfig, path_to_cochleagrams: Path, timestamp: str, elapsed_time: str) -> str:
    # Load cochleagram summary
    with open(glob.glob((path_to_cochleagrams / '_summary_*.txt').as_posix())[0], 'r') as f:
        cochleagram_summary = f.read()

    # TODO: Change summary once freeze training is implemented
    summary = f'##### MODEL TESTING INFO #####\n' \
              f'Timestamp: {timestamp}\n' \
              f'Total elapsed time: {elapsed_time}\n' \
              f'Config:\n{pprint.pformat(eval_config)}\n\n' \
              f'Based on the following cochleagram generation:\n' \
              f'{cochleagram_summary}\n'
    return summary


def predict_with_ground_truth(model, dataset):
    """
    Custom prediction loop to yield model predictions and ground truth labels.

    Args:
        model: The trained model.
        dataset: A tf.data.Dataset yielding (inputs, labels) tuples.

    Yields:
        predictions: The model's predictions for the batch.
        labels: The ground truth labels for the batch.
    """
    for batch in dataset:
        inputs, labels = batch  # Extract inputs and labels from the dataset
        predictions = model(inputs, training=False)  # Perform inference
        yield predictions, labels


if __name__ == '__main__':
    main()

