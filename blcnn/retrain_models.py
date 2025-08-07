import csv
import datetime
import glob
import logging
import pprint
import sys
import time
import traceback
from pathlib import Path

import coloredlogs
import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from blcnn.net_builder import single_example_parser
from blcnn.run_models import predict_with_ground_truth
from blcnn.util import get_unique_folder_name, FreezeTrainingConfig, load_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main() -> None:
    """
    Steps:
    1. Load the pre-trained model
    2. Load the dataset
    3. Freeze all layers except one and retrain
    -> Cut-off? dynamic / fixed?
    4. Save the model (multiple checkpoints?)
    5. Persist a summary containing the training error

    Questions:
    - How long does retraining take? Longer if more layers are trained?
    -> Feasible to do permutation of all layers on all models with frequent checkpoints?

    Plan
    1. One model, one layer, freeze all except last layer, retrain, look at performance
    2. Loop: One layer unfrozen from the back, retrain from scratch each time, look at performance

    - Which loss to use?
        -> What is an acceptable loss and accuracy during training?
    - Look at @tf.function
    - Look at tensorboard
    - Run stuff on lab PC
    - Custom metric to report elevation gain during training? Prob. not possible bc it requires access to many data points
    - Slow! Check if model is using tf Variables everywhere!
        -> Is there an official way to check for redundant calls etc? "Optimizing performance"?
    - Check if gradient checkpointing is needed (trade speed for memory usage). Maybe it's using swap and that's why it's slow.
    - Maybe something in porting the models to TF2 went wrong. Optimizers?
    - The sparse cat. acc. metric works on flattened data thus is expected to be bad -> add own metric which works in 2D space

    Checks
    - Check if input data has correct labels by running the testing on original models (I did that I think)


    Questions
    - What is done in BN layers? What's saved? are they trainable or not in the end?



    """

    retrain_one_model()


def retrain_one_model():
    """
    Dataset stuff
        - Make better train / test split
            - Add another writer
            - Change names of outputs from cochleagrams.tfrecord to train_cochleagrams.tfrecord and test_cochleagrams.tfrecord
            - For each cochleagram flip a coin and add it to either of the datasets
            - Change the summary file: Add another counter so there's 2 counts, one for train and one for test
            - Add split parameter to config.yml that is used for the coin flip chance
        - Make small dataset with ~20 batches to verify the program is working
            - just use dataset.take(20).save() or something.

    Retraining stuff
        - reset model between runs -> Just load it new from disk I guess
        - persist model after each run with correct ngram in name

    Testing stuff
        - pipeline that loads in each retrained version and and tests it on the test set (for both HRTFs, Kemar and new)




        - See if only training one layer decreases the memory needed (compile again before training)
        →  It does! It trains faster (615ms/step compared to 1s/step) and I didn’t get the ‘ran out of memory’ warnings

        - Set up full retraining env; either include testing or if models are small enough, persist them after training
        →  Yes, persist. They’re only 215MB, so in total it’ll be 72*215 = 15.5GB

        sidequest:
        - see if training with kemar cochleagrams starts of with higher precision
        →  loss starts at 2.4, accuracy at 0.37
    """
    config = load_config('blcnn/config.yml')
    hrtf_label = config.freeze_training.labels[0]  # Use the first label for now
    model_id = config.freeze_training.models_to_use[0]  # Use the first model for now

    path_to_cochleagrams = Path(f'data/cochleagrams/{hrtf_label}')
    path_to_model = Path(f'models/keras/net{model_id}.keras')

    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    start_time = time.time()

    # Create destination directory for the retraining run
    dest = get_unique_folder_name(f'models/freeze_trained_{hrtf_label}')
    Path(dest).mkdir(parents=True, exist_ok=False)
    Path(f'data/output/ft_{timestamp}').mkdir(parents=True, exist_ok=True)

    ngrams = extract_layer_ngrams_indices(path_to_model)
    tqdm.write(f'n-grams extracted from {path_to_model}: {ngrams}')

    dataset_length = None
    file_name = glob.glob((path_to_cochleagrams / '_summary_*.txt').as_posix())[0]
    for line in open(file_name, 'r'):
        if 'Train dataset size (nr of cochleagrams):' in line:
            dataset_length = int(line.split(': ')[1].strip())
            break

    try:
        for ngram in tqdm(ngrams, desc='Retraining models', unit='ngram', position=0, leave=True):
            tqdm.write(f'Retraining ngram: {ngram}')
            retraining_run(ngram, path_to_model, path_to_cochleagrams, dataset_length, dest, timestamp)
            tqdm.write('\n' + '=' * 50 + '\n')

    except Exception as e:
        logger.error(f'An error occurred during freeze training: {e}\n'
                     f'{traceback.print_exc()}')

    finally:
        total_elapsed_time = str(datetime.timedelta(seconds=time.time() - start_time))
        summary = summarize_freeze_training_info(config.freeze_training, hrtf_label, timestamp, total_elapsed_time, dest)
        with open(dest / f'_summary_{timestamp}.txt', 'w') as f:
            f.write(summary)


def retraining_run(ngram: list, path_to_model: Path, path_to_cochleagrams: Path, dataset_length: int, dest: Path,
                   timestamp: str) -> None:
    """
    Retrain the model on the given ngram.
    Dataset and model are loaded here, so they are not reused in between runs.
    """
    start_time_single_run = time.time()

    # Load model
    model = keras.models.load_model(path_to_model)

    # Load dataset
    train_dataset = (
        tf.data.TFRecordDataset(path_to_cochleagrams / 'train_cochleagrams.tfrecord', compression_type="GZIP")
        .map(lambda serialized_example: single_example_parser(serialized_example))
        # .shuffle(64)
        .batch(16, drop_remainder=True)
        .prefetch(1)  # Lets the CPU prepare the next batch while the GPU is still busy with the current one
    )

    # Freeze all layers
    for layer in model.layers:
        layer.trainable = False

    # Unfreeze the layers specified in the ngram
    for i in ngram:
        model.layers[i].trainable = True

    tqdm.write(f'Trainable layers: {[model.layers[i].name for i in ngram]}')

    # Compile the model
    model.compile(optimizer=keras.optimizers.legacy.Adam(), loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    # dest = Path(f'data/output/freeze_trained_{timestamp}')
    ngram_repr = '_'.join(str(layer_id) for layer_id in ngram)

    # # Logging
    # log_path = dest / 'logs'
    # log_path.mkdir(parents=True, exist_ok=True)
    # # Log directory should include: [hrtf label] / [network ID, trainable layers, (timestamp)}
    # log_dir = log_path / ngram_repr
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1, update_freq=1000)
    #
    # # Train the model
    # model.fit(train_dataset, epochs=1, callbacks=[tensorboard_callback], verbose=1,
    #           steps_per_epoch=dataset_length // 16)
    #
    # # Save the model
    # model.save(dest / f'net1_{ngram_repr}.keras')

    elapsed_time = str(datetime.timedelta(seconds=time.time() - start_time_single_run))
    with open(dest / f'elapsed_time_per_run.txt', 'a') as f:
        f.write(f'{elapsed_time} ({ngram_repr}) --- {[model.layers[i].name for i in ngram]}\n')


def summarize_freeze_training_info(freeze_training_config: FreezeTrainingConfig,
                                   hrtf_label: str,
                                   timestamp: str,
                                   total_elapsed_time: str,
                                   dest: Path) -> str:
    # Load cochleagram summary
    path_to_cochleagrams = Path(f'data/cochleagrams/{hrtf_label}')
    with open(glob.glob((path_to_cochleagrams / '_summary_*.txt').as_posix())[0], 'r') as f:
        cochleagram_summary = f.read()

    summary = f'##### FREEZE TRAINING INFO #####\n' \
              f'HRTF label: {hrtf_label}\n' \
              f'Timestamp: {timestamp}\n\n' \
              f'Total elapsed time: {total_elapsed_time}\n' \
              f'Config:\n{pprint.pformat(freeze_training_config)}\n\n' \
              f'Based on the following BRIR generation:\n' \
              f'{cochleagram_summary}\n\n' \
              f'################################\n' \
              f'Models saved to: {dest}\n' \
              f'################################\n\n'
    return summary


# def other_stuff():
#     # Predict
#     true_classes = []
#     pred_classes = []
#
#     # Evaluate the model
#     for predictions, labels in tqdm(predict_with_ground_truth(model, dataset), total=int(dataset_length * 0.8) // 16,
#                                     unit='batches'):
#         # for predictions, labels in predict_with_ground_truth(model, dataset):
#         true_classes.append(labels.numpy())
#         pred_classes.append(predictions.numpy())
#
#     true_classes = np.concatenate(true_classes, axis=0)
#     pred_classes = np.concatenate(pred_classes, axis=0).argmax(axis=1)
#
#     # write to CSV
#     with open(dest / f'{ngram_repr}.csv', 'w', newline='') as f:
#         writer = csv.writer(f)
#         # writer.writerow(['true_class', 'pred_class', 'stim_name'])
#         writer.writerow(['true_class', 'pred_class'])
#         # writer.writerows(zip(true_classes, pred_classes, stim_names))
#         writer.writerows(zip(true_classes, pred_classes))
#
#     # TODO: Extract relevant code from plotting.py for elev gain
#     # -> Write function that loads the CSVs and plots the elev gain into matrix
#     # TODO: Get better sound dataset.


def extract_layer_ngrams_indices(path_to_model: Path) -> list:
    """
    Take a model and return a list containing the indices of ngrams for each conv2d layer.
    Additionally return those ngrams with the Dense layer added.
    """
    model = keras.models.load_model(path_to_model)
    ngram_indices = []
    conv2d_indices = []

    for i, layer in enumerate(model.layers):
        print('Layer:', i, layer.name)
        if isinstance(layer, keras.layers.Conv2D):
            conv2d_indices.append(i)
        if layer.name == 'dense':
            dense_index = i

    print('Conv2D indices:', conv2d_indices)
    print('Dense index:', dense_index)

    # Get the ngram indices
    for i in range(len(conv2d_indices)):
        for j in range(len(conv2d_indices)):
            if conv2d_indices[i:j + 1]:
                ngram_indices.append(conv2d_indices[i:j + 1])

    ngram_indices.sort(key=lambda x: len(x))

    # Add a copy of each ngram with the dense layer added
    for i in range(len(ngram_indices)):
        ngram_indices.append(ngram_indices[i] + [dense_index])

    return ngram_indices


if __name__ == '__main__':
    main()
