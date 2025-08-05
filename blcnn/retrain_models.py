import csv
import datetime
import glob
import sys
import time
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from blcnn.net_builder import single_example_parser
from blcnn.run_models import predict_with_ground_truth


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
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    Path(f'data/output/ft_{timestamp}').mkdir(parents=True, exist_ok=True)

    path_to_model = Path('models/keras/net1.keras')
    # path_to_cochleagrams = Path(f'data/cochleagrams/francl_data_transformed_concatenated/')
    path_to_cochleagrams = Path(f'data/cochleagrams/naturalsounds165_hrtf_nh2')

    model = keras.models.load_model(path_to_model)
    # TODO: Add elev gain metric?
    # If not possible as a metric, add it after model.fit()

    # Extract ngrams
    # for loop through ngrams w/tqdm
    # load dataset -> make split, ideally shuffle
    # freeze layers, compile, add callback (w/ unfrozen layers as folder name), fit
    # Evaluate elev gain on test set -> Append to file

    ngrams = extract_layer_ngrams_indices(model)
    print('Ngrams:', ngrams)

    # dataset1 = (
    #     tf.data.TFRecordDataset(path_to_cochleagrams / 'cochleagrams.tfrecord', compression_type="GZIP")
    #     .map(lambda serialized_example: single_example_parser(serialized_example))
    #     .shuffle(64)
    #     .batch(16, drop_remainder=True)
    #     .prefetch(1)  # Lets the CPU prepare the next batch while the GPU is still busy with the current one
    # )
    # dataset_length = 0
    # for _ in dataset1:
    #     dataset_length += 1

    total_samples = None
    file_name = glob.glob((path_to_cochleagrams / '_summary_*.txt').as_posix())[0]
    for line in open(file_name, 'r'):
        if 'Number of cochleagrams generated:' in line:
            total_samples = int(line.split(': ')[1].strip())
            break

    for ngram in tqdm(ngrams):
        tqdm.write(f'Processing ngram: {ngram}')



        # Retrain model on ngram
        retraining_run(ngram, model, path_to_cochleagrams, total_samples, timestamp)


def retraining_run(ngram: list, model: keras.Model, path_to_cochleagrams: Path, dataset_length: int, timestamp: str) -> None:
    """
    Retrain the model on the given ngram.
    """

    # Load dataset
    dataset = (
        tf.data.TFRecordDataset(path_to_cochleagrams / 'cochleagrams.tfrecord', compression_type="GZIP")
        .map(lambda serialized_example: single_example_parser(serialized_example))
        # .shuffle(64)
        .batch(16, drop_remainder=True)
        .prefetch(1)  # Lets the CPU prepare the next batch while the GPU is still busy with the current one
    )

    # Freeze all layers except the last one
    # for layer in model.layers:
    #     layer.trainable = False
    #
    # for i in ngram:
    #     model.layers[i].trainable = True

    # for layer in model.layers:
    #     if layer.trainable:
    #         print('Trainable:', layer.name)
    #     else:
    #         print('Frozen:   ', layer.name)


    train_dataset, test_dataset = keras.utils.split_dataset(dataset, left_size=0.8)
    print('Training split:', dataset_length * 0.8)
    # train_dataset = dataset.take(training_split)
    # test_dataset = dataset.skip(training_split)


    # Compile the model
    model.compile(optimizer=keras.optimizers.legacy.Adam(), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    # Create destination directory for the retraining run
    # TODO: Change dest to save test results -> one folder with timestamp, then one file per training run w/ ngram info. In name?
    dest = Path(f'data/output/ft_{timestamp}')
    ngram_repr = '_'.join(str(layer_id) for layer_id in ngram)

    # Logging
    log_path = Path(f'data/logs/ft_{timestamp}')
    log_path.mkdir(parents=True, exist_ok=True)
    # Log directory should include: [hrtf label] / [network ID, trainable layers, (timestamp)}
    log_dir = log_path / "fit" / ngram_repr
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1, update_freq=1000)

    # Train the model
    model.fit(train_dataset, epochs=1, callbacks=[tensorboard_callback], verbose=1)#, steps_per_epoch=training_split // 16)

    # Predict
    true_classes = []
    pred_classes = []

    # Evaluate the model
    for predictions, labels in tqdm(predict_with_ground_truth(model, dataset), total=int(dataset_length * 0.8) // 16, unit='batches'):
    # for predictions, labels in predict_with_ground_truth(model, dataset):
        true_classes.append(labels.numpy())
        pred_classes.append(predictions.numpy())

    true_classes = np.concatenate(true_classes, axis=0)
    pred_classes = np.concatenate(pred_classes, axis=0).argmax(axis=1)

    # write to CSV
    with open(dest / f'{ngram_repr}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(['true_class', 'pred_class', 'stim_name'])
        writer.writerow(['true_class', 'pred_class'])
        # writer.writerows(zip(true_classes, pred_classes, stim_names))
        writer.writerows(zip(true_classes, pred_classes))


    # TODO: Extract relevant code from plotting.py for elev gain
    # -> Write function that loads the CSVs and plots the elev gain into matrix
    # TODO: Get better sound dataset.



# @tf.function
def retrain_last_layer():
    # hrtf_label = 'slab_kemar'
    # hrtf_label = 'hrtf_nh2'

    path_to_models = Path('models/keras/')
    path_to_cochleagrams = Path(f'data/cochleagrams/uso_500ms_raw_slab_kemar_anechoic/')

    model = keras.models.load_model(path_to_models / 'net1.keras')




    # sys.exit()

    # dataset = (
    #     tf.data.TFRecordDataset(path_to_cochleagrams / 'cochleagrams.tfrecord', compression_type="GZIP")
    #     .map(lambda serialized_example: single_example_parser(serialized_example))
    #     .shuffle(64)
    #     .batch(16, drop_remainder=True)
    #     .prefetch(1)  # Lets the CPU prepare the next batch while the GPU is still busy with the current one
    # )



    # Freeze all layers except the last one
    # for layer in model.layers[:-1]:
    for layer in model.layers:
        layer.trainable = False

    model.layers[-10].trainable = True

    for layer in model.layers:
        if layer.trainable:
            print('Trainable:', layer.name)
        else:
            print('Frozen:   ', layer.name)

    ngrams = extract_layer_ngrams_indices(model)
    print('Ngrams:', ngrams)
    sys.exit()

    # Compile the model
    model.compile(optimizer=keras.optimizers.legacy.Adam(), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    log_path = Path("data/logs/uso_500ms_raw_slab_kemar_anechoic")
    log_path.mkdir(parents=True, exist_ok=True)
    # Log directory should include: [hrtf label] / [network ID, trainable layers, (timestamp)}
    log_dir = log_path / "fit" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1, update_freq=1000)


    # Train the model
    model.fit(dataset, epochs=1, callbacks=[tensorboard_callback])


def extract_layer_ngrams_indices(model: keras.Model) -> list:
    """
    Take a model and return a list containing the indices of ngrams for each conv2d layer.
    Additionally return those ngrams with the Dense layer added.
    """
    ngram_indices = []
    conv2d_indices = []

    for i, layer in enumerate(model.layers):
        print ('Layer:', i, layer.name)
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