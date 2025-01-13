import csv
import datetime
import glob
import logging
import time
from pathlib import Path

import coloredlogs
import keras
import numpy as np
import tensorflow as tf

from util import get_unique_folder_name
from net_builder import single_example_parser

logger = tf.get_logger()
logger.setLevel(logging.INFO)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main() -> None:
    # Disable GPU
    logger.info(f'Physical devices: {tf.config.list_physical_devices()}')
    only_cpu = False
    if only_cpu:
        tf.config.set_visible_devices([], 'GPU')
        logger.info('Removing GPU devices to force CPU usage')
    logger.info(f'Visible devices: {tf.config.get_visible_devices()}')

    test_models()


def test_models():
    start_time = time.time()
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    path_to_models = Path('models/keras/')
    path_to_cochleagrams = Path('data/cochleagrams/slab_default_kemar')

    dest = get_unique_folder_name(f'data/output/{path_to_cochleagrams.name}/')
    Path(dest).mkdir(parents=True, exist_ok=False)

    for i in range(1, 11):
        model_path = path_to_models / f'net{i}.keras'
        test_model_and_persist_data(model_path, path_to_cochleagrams, dest)

    elapsed_time = str(datetime.timedelta(seconds=time.time() - start_time))
    summary = summarize_testing(path_to_models, path_to_cochleagrams, timestamp, elapsed_time)
    logger.info(summary)
    with open(dest / f'_summary_{timestamp}.txt', 'w') as f:
        f.write(summary)


def test_model_and_persist_data(model_path: Path = None, path_to_cochleagrams: Path = None, dest: Path = None):
    logger.info(f'Testing model: {model_path}, with cochleagrams from: {path_to_cochleagrams}, saving to: {dest}')

    model = keras.models.load_model(model_path)

    # TODO: Better performance by batching Example protos and using parse_example; see if useful
    dataset = (
        tf.data.TFRecordDataset(path_to_cochleagrams / 'cochleagrams.tfrecord', compression_type="GZIP")
        .map(lambda serialized_example: single_example_parser(serialized_example))
        # .shuffle(64)
        .batch(16, drop_remainder=True)
        .prefetch(1)
        .take(20)
    )

    # Predict
    pred_classes = []
    true_classes = []

    for predictions, labels in predict_with_ground_truth(model, dataset):
        pred_classes.append(predictions.numpy())
        true_classes.append(labels.numpy())

    pred_classes = np.concatenate(pred_classes, axis=0)
    true_classes = np.concatenate(true_classes, axis=0)

    # pred_classes, true_classes = model.predict(dataset)
    print('Pred classes shape:', pred_classes.shape)
    print('True classes shape:', true_classes.shape)
    print('Pred classes:', pred_classes)
    print('True classes:', true_classes)

    # Argmax to get the predicted class
    pred_classes = np.argmax(pred_classes, axis=1)

    print('pred_classes = [', *pred_classes, ']', sep=', ')
    print('true_classes =', *true_classes)

    # write to CSV
    with open(dest / f'{model_path.name.split(".")[0]}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true_class', 'pred_class'])
        writer.writerows(zip(true_classes, pred_classes))


def summarize_testing(model_path: Path, path_to_cochleagrams: Path, timestamp: str, elapsed_time: str) -> str:
    # Load cochleagram summary
    with open(glob.glob((path_to_cochleagrams / '_summary_*.txt').as_posix())[0], 'r') as f:
        cochleagram_summary = f.read()

    # TODO: Change summary once freeze training is implemented
    summary = f'##### MODEL TESTING INFO #####\n' \
              f'Timestamp: {timestamp}\n' \
              f'Total elapsed time: {elapsed_time}\n' \
              f'Loaded model(s) from: {model_path}\n\n' \
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

