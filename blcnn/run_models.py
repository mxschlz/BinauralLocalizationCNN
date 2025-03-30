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
from tqdm import tqdm

from util import get_unique_folder_name, load_config, RunModelsConfig
from net_builder import single_example_parser

logger = tf.get_logger()
logger.setLevel(logging.INFO)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main() -> None:
    """
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
    eval_config = load_config('blcnn/config.yml').run_models
    logger.info(f'Loaded config: {eval_config}')

    for label in eval_config.labels:
        test_multiple_models(label, eval_config)


def test_multiple_models(label: str, run_models_config: RunModelsConfig) -> None:
    """
    Run the testing for one HRTF label using the models specified in the config.
    """
    start_time = time.time()
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')

    path_to_models = Path('models/keras/')
    path_to_cochleagrams = Path(f'data/cochleagrams/{label}')

    dest = get_unique_folder_name(f'data/output/{path_to_cochleagrams.name}/')
    Path(dest).mkdir(parents=True, exist_ok=False)

    for model_id in run_models_config.models_to_use:
        model_path = path_to_models / f'net{model_id}.keras'
        test_single_model(model_path, path_to_cochleagrams, dest)

    elapsed_time = str(datetime.timedelta(seconds=time.time() - start_time))
    summary = summarize_testing(run_models_config, path_to_cochleagrams, timestamp, elapsed_time)
    logger.info(summary)
    with open(dest / f'_summary_{timestamp}.txt', 'w') as f:
        f.write(summary)


def test_single_model(model_path: Path = None, path_to_cochleagrams: Path = None, dest: Path = None):
    """
    Test a single model with the given cochleagrams and save the results to a CSV file.
    """
    logger.info(f'Testing model: {model_path}, with cochleagrams from: {path_to_cochleagrams}, saving to: {dest}')

    model = keras.models.load_model(model_path)

    nr_examples = sum(1 for _ in tqdm(tf.data.TFRecordDataset(path_to_cochleagrams / 'cochleagrams.tfrecord', compression_type="GZIP"), unit='examples', desc='Counting examples'))
    # nr_examples = tf.data.TFRecordDataset(path_to_cochleagrams / 'cochleagrams.tfrecord', compression_type="GZIP").reduce(np.int64(0), lambda x, _: x + 1)
    logger.info(f'Number of examples in dataset: {nr_examples}')

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
    # stim_names = []

    for predictions, labels in tqdm(predict_with_ground_truth(model, dataset), total=nr_examples/16, unit='batches'):
    # for predictions, labels, names in tqdm(predict_with_ground_truth(model, dataset), unit='batches'):
        true_classes.append(labels.numpy())
        pred_classes.append(predictions.numpy())
        # stim_names.append(names.numpy())

    true_classes = np.concatenate(true_classes, axis=0)
    pred_classes = np.concatenate(pred_classes, axis=0).argmax(axis=1)
    # stim_names = np.concatenate(stim_names, axis=0)
    # stim_names = [name.decode('utf-8') for name in stim_names]

    # write to CSV
    with open(dest / f'{model_path.name.split(".")[0]}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(['true_class', 'pred_class', 'stim_name'])
        writer.writerow(['true_class', 'pred_class'])
        # writer.writerows(zip(true_classes, pred_classes, stim_names))
        writer.writerows(zip(true_classes, pred_classes))


def summarize_testing(run_models_config: RunModelsConfig, path_to_cochleagrams: Path, timestamp: str, elapsed_time: str) -> str:
    # Load cochleagram summary
    with open(glob.glob((path_to_cochleagrams / '_summary_*.txt').as_posix())[0], 'r') as f:
        cochleagram_summary = f.read()

    # TODO: Change summary once freeze training is implemented
    summary = f'##### MODEL TESTING INFO #####\n' \
              f'Timestamp: {timestamp}\n' \
              f'Total elapsed time: {elapsed_time}\n' \
              f'Config:\n{pprint.pformat(run_models_config)}\n\n' \
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
        # inputs, labels, names = batch  # Extract inputs and labels from the dataset
        predictions = model(inputs, training=False)  # Perform inference
        yield predictions, labels


if __name__ == '__main__':
    main()

