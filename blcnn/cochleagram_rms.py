from pathlib import Path
import numpy as np
import tensorflow as tf

from blcnn.net_builder import single_example_parser


def compute_mean_cochleagram_rms(path_to_cochleagrams):

    dataset = (
        tf.data.TFRecordDataset(path_to_cochleagrams, compression_type="GZIP")
        .map(lambda serialized_example: single_example_parser(serialized_example))
        .take(100)
    )
    rms_values = []
    for cochleagram, _ in dataset:
        cochleagram_np = cochleagram.numpy()
        rms = np.sqrt(np.mean(cochleagram_np**2))
        rms_values.append(rms)

    mean_rms = np.mean(rms_values)
    std_rms = np.std(rms_values)
    return mean_rms, std_rms


if __name__ == "__main__":
    # all .tfrecord files in subfolders of data/cochleagrams/


    for path in Path('data/cochleagrams/').rglob('*.tfrecord'):
        rms = compute_mean_cochleagram_rms(path)
        print(f'Path: {path}, Mean RMS: {rms[0]:.3f}, Std RMS: {rms[1]:.3f}')