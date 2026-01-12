import csv
import datetime
import glob
import logging
import pprint
import sys
import time
import traceback
from pathlib import Path
from typing import List
import multiprocessing as mp
import pickle

import coloredlogs

from blcnn.net_builder import single_example_parser
from blcnn.run_models import predict_with_ground_truth
from blcnn.util import get_unique_folder_name, FreezeTrainingConfig, load_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
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

    # retrain_one_model()
    # print("Starting isolated retrain case with learning rate 0.005")
    # retrain_case_isolated(0.005)
    # print("-----")
    # print("Starting isolated retrain case with learning rate 0.0005")
    # retrain_case_isolated(0.0005)

    # persist_and_reload_model()
    # test_if_retraining_works()
    # test_logging()
    execute_retrain()


def persist_and_reload_model():
    # Test if model persistence and reloading works correctly after retraining
    import tensorflow as tf
    import keras
    from keras import backend as K
    K.clear_session()

    model: keras.Model = keras.models.load_model('models/keras_momentum_9e-1/net1.keras', compile=False)
    model.compile(optimizer=keras.optimizers.legacy.Adam(1e-3), loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    model.save('models/keras_momentum_9e-1/net1_test_persist.keras')

    # Clear session again
    K.clear_session()
    model_reloaded: keras.Model = keras.models.load_model('models/keras_momentum_9e-1/net1_test_persist.keras',
                                                          compile=False)
    model_reloaded.compile(optimizer=keras.optimizers.legacy.Adam(1e-3), loss='sparse_categorical_crossentropy',
                           metrics=['sparse_categorical_accuracy'])

    # -> This works without error or warning


def check_if_retraining_works():
    """
    - Now test funtionality
        - Test model w/ nh2 test set
        - Train model on nh2 train data
        - Test again
        - Persist and reload
        - Test again
    """
    import tensorflow as tf
    import keras
    from keras import backend as K
    K.clear_session()
    model: keras.Model = keras.models.load_model('models/keras_momentum_9e-1/net1.keras', compile=False)
    train_dataset = (
        # tf.data.TFRecordDataset('data/cochleagrams/francl_data_transformed_concatenated/test_cochleagrams.tfrecord', compression_type="GZIP")
        tf.data.TFRecordDataset('data/cochleagrams/naturalsounds165_hrtf_nh2_20/train_cochleagrams.tfrecord',
                                compression_type="GZIP")
        .map(lambda serialized_example: single_example_parser(serialized_example))
        .shuffle(64)
        .batch(16, drop_remainder=True)
        # .take(830)
        .repeat()
        .prefetch(1)
    )
    for layer in model.layers:  # Freeze all layers
        layer.trainable = False
    for i in [34]:  # Unfreeze specified layers
        model.layers[i].trainable = True
    logging.info(f'Trainable layers: {[layer.name for layer in model.layers if layer.trainable]}')
    model.compile(optimizer=keras.optimizers.legacy.Adam(1e-3), loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    infer_model_online(model, Path('data/cochleagrams/naturalsounds165_hrtf_nh2_20'), [0], Path('data/output/naturalsounds165_hrtf_nh2_test_before_retrain'), tf)

    model.fit(train_dataset, epochs=10, verbose=1, steps_per_epoch=13288 // 16)

    infer_model_online(model, Path('data/cochleagrams/naturalsounds165_hrtf_nh2_20'), [1], Path('data/output/naturalsounds165_hrtf_nh2_test_retrain'), tf)


    model.save('models/keras_momentum_9e-1/net1_test_retrain.keras')

    # Clear session again
    K.clear_session()
    model_reloaded: keras.Model = keras.models.load_model('models/keras_momentum_9e-1/net1_test_retrain.keras', compile=False)
    model_reloaded.compile(optimizer=keras.optimizers.legacy.Adam(1e-3), loss='sparse_categorical_crossentropy',
                            metrics=['sparse_categorical_accuracy'])
    infer_model_online(model_reloaded, Path('data/cochleagrams/naturalsounds165_hrtf_nh2_20'), [2], Path('data/output/naturalsounds165_hrtf_nh2_test_retrain_reloaded'), tf)
    # -> Works!!


def check_logging():
    import tensorflow as tf
    import keras
    from keras import backend as K
    K.clear_session()
    model: keras.Model = keras.models.load_model('models/keras_momentum_9e-1/net1.keras')

    train_dataset = (
        tf.data.TFRecordDataset('data/cochleagrams/naturalsounds165_hrtf_nh2_20/train_cochleagrams.tfrecord',
                                compression_type="GZIP")
        .map(lambda serialized_example: single_example_parser(serialized_example))
        .shuffle(64)
        .batch(16, drop_remainder=True)
        .repeat()
        .prefetch(1)
    )

    for layer in model.layers:
        layer.trainable = False

    for i in [34]:
        model.layers[i].trainable = True

    logging.info(f'Trainable layers: {[layer.name for layer in model.layers if layer.trainable]}')

    model.compile(optimizer=keras.optimizers.legacy.Adam(0.001), loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="data/logs/retrain_test",
                                                       histogram_freq=1,
                                                       write_graph=True,
                                                       write_images=True,
                                                       write_steps_per_second=True,
                                                       update_freq='batch',)

    model.fit(train_dataset, epochs=10, callbacks=[tensorboard_callback], verbose=1, steps_per_epoch=13288 // 16)


def retrain_case_isolated(learning_rate: float) -> None:
    import tensorflow as tf
    import keras
    from keras import backend as K

    # Optional but recommended for reproducibility:
    # Clear everything
    K.clear_session()
    # Check if clear worked
    # if K.get_session() is not None:
    #     logging.warning("Keras session not cleared properly.")
    model: keras.Model = keras.models.load_model('models/keras_momentum_9e-1/net1.keras')

    train_dataset = (
        # tf.data.TFRecordDataset('data/cochleagrams/francl_data_transformed_concatenated/test_cochleagrams.tfrecord', compression_type="GZIP")
        tf.data.TFRecordDataset('data/cochleagrams/naturalsounds165_hrtf_nh2_20/train_cochleagrams.tfrecord',
                                compression_type="GZIP")
        .map(lambda serialized_example: single_example_parser(serialized_example))
        .shuffle(64)
        .batch(16, drop_remainder=True)
        # .take(830)
        .repeat()
        .prefetch(1)
    )

    for layer in model.layers:  # Freeze all layers
        layer.trainable = False

    for i in [34]:  # Unfreeze specified layers
        model.layers[i].trainable = True

    logging.info(f'Trainable layers: {[layer.name for layer in model.layers if layer.trainable]}')

    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate), loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    model.fit(train_dataset, epochs=10, verbose=1, steps_per_epoch=13288 // 16)
    # model.fit(train_dataset, epochs=1, callbacks=[tensorboard_callback], verbose=1, steps_per_epoch=dataset_length // 16)

    # test_model_online(model, Path('data/cochleagrams/naturalsounds165_hrtf_nh2_20'), [34], Path('data/output/naturalsounds165_hrtf_nh2_isolated_case'), tf)


def _worker_retrain(args, conn):
    # This code runs in a separate, clean Python process
    try:
        # Unpack
        (path_to_coch, path_to_model, layers_to_train, train_len, dest) = args

        # Now call your real training function
        result = retrain(path_to_coch, path_to_model, layers_to_train, train_len, dest)

        conn.send(("ok", result))
    except Exception as e:
        import traceback
        conn.send(("err", traceback.format_exc()))
    finally:
        conn.close()


def run_retrain_isolated(path_to_coch, path_to_model, layers_to_train, train_len, dest):
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(
        target=_worker_retrain,
        args=((path_to_coch, path_to_model, layers_to_train, train_len, dest), child_conn),
    )
    p.start()
    status, payload = parent_conn.recv()
    p.join()

    if status == "err":
        raise RuntimeError("Retraining subprocess failed:\n" + payload)

    return payload


def execute_retrain():
    """
    Load config and execute retraining runs based on that.
    """
    config = load_config('blcnn/config.yml')
    # hrtf_labels = config.freeze_training.labels  # Use the first label for now
    hrtf_labels = ['naturalsounds165_hrtf_nh2_20']
    # model_ids = config.freeze_training.models_to_use  # Use the first model for now
    model_ids = [1]
    ngram_lengths = config.freeze_training.ngrams

    for hrtf_label in hrtf_labels:
        path_to_cochleagrams = Path(f'data/cochleagrams/{hrtf_label}')

        # Get train dataset length from summary file
        train_dataset_length = None
        file_name = glob.glob((path_to_cochleagrams / '_summary_*.txt').as_posix())[0]
        for line in open(file_name, 'r'):
            if 'Train dataset size (nr of cochleagrams):' in line:
                train_dataset_length = int(line.split(': ')[1].strip())
                break

        # Create destination base path to store testing output for this HRTF
        dest = get_unique_folder_name(f'data/ft/{path_to_cochleagrams.name}')

        for model_id in model_ids:
            path_to_model = Path(f'models/keras_momentum_9e-1/net{model_id}.keras')

            # Compute n-grams for this model
            ngrams = compute_layer_ngrams_indices(Path('models/keras_momentum_9e-1/layer_indices.txt'), model_id, ngram_lengths)
            logger.info(f'Computed n-grams for model ID {model_id}: {ngrams}')

            for layers_to_train in ngrams:
                logger.info(f'Starting retraining for HRTF: {hrtf_label}, Model ID: {model_id}, Layers to train: {layers_to_train}')
                current_dest = dest / Path(f'ft_{"_".join(str(l) for l in layers_to_train)}')
                current_dest.mkdir(parents=True, exist_ok=True)
                logger.info(f'Saving to: {current_dest}')
                try:
                    retrain(path_to_cochleagrams, path_to_model, layers_to_train, train_dataset_length, current_dest)
                except Exception as e:
                    logger.error(f'Error during retraining for HRTF: {hrtf_label}, Model ID: {model_id}, Layers to train: {layers_to_train}')
                    logger.error(traceback.format_exc())
                logger.info('\n\n')



def retrain(path_to_cochleagrams: Path, path_to_model: Path, layers_to_train: List[int], train_dataset_length: int,
            dest_base_path: Path) -> None:
    """
    Retrains the given model on the given cochleagrams, only training the specified layers.
    """
    import tensorflow as tf
    import keras
    from keras import backend as K
    K.clear_session()

    model: keras.Model = keras.models.load_model(path_to_model, compile=False)
    train_dataset = (
        tf.data.TFRecordDataset(path_to_cochleagrams / 'train_cochleagrams.tfrecord', compression_type="GZIP")
        .map(lambda serialized_example: single_example_parser(serialized_example))
        .shuffle(64)
        .batch(16, drop_remainder=True)
        .repeat()
        .prefetch(1)
    )

    for layer in model.layers:  # Freeze all layers
        layer.trainable = False

    for i in layers_to_train:  # Unfreeze specified layers
        model.layers[i].trainable = True

    logging.info(f'Trainable layers: {[layer.name for layer in model.layers if layer.trainable]}')

    model.compile(optimizer=keras.optimizers.legacy.Adam(1e-3), loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=(dest_base_path / Path('logs')).as_posix(),
                                                         histogram_freq=1,
                                                         write_graph=True,
                                                         write_images=True,
                                                         write_steps_per_second=True,
                                                         update_freq='batch')

    model.fit(train_dataset, epochs=10, callbacks=[tensorboard_callback], verbose=1, steps_per_epoch=train_dataset_length // 16)
    # model.fit(train_dataset, epochs=1, callbacks=[tensorboard_callback], verbose=1, steps_per_epoch=1)

    model.save(dest_base_path / Path('model.keras'))

    infer_model_online(model, path_to_cochleagrams, layers_to_train, dest_base_path, tf)


def infer_model_online(model, path_to_cochleagrams: Path, layers_to_train: List[int], dest: Path, tf) -> None:
    import numpy as np
    from tqdm import tqdm
    # ngram_repr = '_'.join(str(layer_id) for layer_id in layers_to_train)
    # Path(dest).mkdir(parents=True, exist_ok=False)

    dataset = (
        tf.data.TFRecordDataset(path_to_cochleagrams / 'test_cochleagrams.tfrecord', compression_type="GZIP")
        .map(lambda serialized_example: single_example_parser(serialized_example))
        .batch(16, drop_remainder=True)
        .prefetch(1)
    )

    # Predict
    true_classes = []
    pred_classes = []
    for predictions, labels in tqdm(predict_with_ground_truth(model, dataset), unit='batches'):
        true_classes.append(labels.numpy())
        pred_classes.append(predictions.numpy())

    true_classes = np.concatenate(true_classes, axis=0)
    pred_classes = np.concatenate(pred_classes, axis=0).argmax(axis=1)

    # write to CSV
    logger.info(f'Writing predictions to CSV in {dest / path_to_cochleagrams.name.split(".")[0]}_directsave.csv')
    with open(dest / 'predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true_class', 'pred_class'])
        writer.writerows(zip(true_classes, pred_classes))


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


def compute_layer_ngrams_indices(path_to_indices: Path, net_id: int, ngram_lengths: List[int]) -> list:
    """
    Take a model and return a list containing the indices of ngrams for each conv2d layer.
    Additionally return those ngrams with the Dense layer added.
    """
    with open(path_to_indices, 'r') as f:
        layer_indices = eval(f.read())
    logger.info(f'Loaded layer indices from models/keras/layer_indices.txt: {layer_indices}')

    conv2d_indices = layer_indices[f'net{net_id}']['conv2d']
    dense_index = layer_indices[f'net{net_id}']['dense']

    ngram_indices = []
    # Get the ngram indices
    for i in range(len(conv2d_indices)):
        for j in range(len(conv2d_indices)):
            if conv2d_indices[i:j + 1]:
                if (j - i + 1) in ngram_lengths:
                    ngram_indices.append(conv2d_indices[i:j + 1])

    # Sort the ngrams by length (shortest first) and then by last index (largest last index first)
    # This way we train from the back and start with small ngrams
    ngram_indices.sort(key=lambda x: (len(x), -x[-1]))

    # Add a copy of each ngram with the dense layer added
    for i in range(len(ngram_indices)):
        ngram_indices.append(ngram_indices[i] + [dense_index])

    # Add dense layer only as well
    ngram_indices.insert(0, [dense_index])

    return ngram_indices


def experiment_alpha():
    """
    Retrain w/ multiple different HRTFs and do testing with those and KEMAR before and after retraining
    For now: Use nh2 model that w/ Dense layer retrained.
    """
    import tensorflow as tf
    import keras
    from keras import backend as K
    K.clear_session()

    dest = get_unique_folder_name('data/output_experiment_alpha')

    # Load model before retraining and after retraining
    model_before: keras.Model = keras.models.load_model('models/keras_momentum_9e-1/net1.keras', compile=False)
    model_after: keras.Model = keras.models.load_model('data/ft/naturalsounds165_hrtf_nh2_20/ft_34/model.keras', compile=False)

    infer_model_online(model_before, Path('data/cochleagrams/naturalsounds165_slab_kemar'), [34], dest / Path('kemar_before_retrain'), tf)
    infer_model_online(model_after, Path('data/cochleagrams/naturalsounds165_slab_kemar'), [34], dest / Path('kemar_after_retrain'), tf)
    infer_model_online(model_before, Path('data/cochleagrams/naturalsounds165_hrtf_nh2_20'), [34], dest / Path('nh2_before_retrain'), tf)
    infer_model_online(model_after, Path('data/cochleagrams/naturalsounds165_hrtf_nh2_20'), [34], dest / Path('nh2_after_retrain'), tf)








if __name__ == '__main__':
    main()

"""
    - See if only training one layer decreases the memory needed (compile again before training)
    →  It does! It trains faster (615ms/step compared to 1s/step) and I didn’t get the ‘ran out of memory’ warnings

    - Set up full retraining env; either include testing or if models are small enough, persist them after training
    →  Yes, persist. They’re only 215MB, so in total it’ll be 72*215 = 15.5GB

    sidequest:
    - see if training with kemar cochleagrams starts of with higher precision
    →  loss starts at 2.4, accuracy at 0.37
"""
