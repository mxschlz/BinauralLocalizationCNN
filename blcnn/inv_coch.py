import os

import numpy as np
import slab
import tensorflow as tf

from net_builder import single_example_parser
from pycochleagram.cochleagram import invert_cochleagram

# dataset_name = 'cochleagrams_2024-12-17_19-14-28/cochs_hrtf_slab_default_kemar.tfrecord'
dataset_name = 'train0.tfrecord'
print(os.getcwd())
i=0
for coch in tf.data.TFRecordDataset(f'../data/processed/{dataset_name}', compression_type="GZIP").map(lambda serialized_example: single_example_parser(serialized_example)):
        # i += 1
        # if not i % 10 == 0:
        #         print(i)
        #         continue
        #
        # else:
        #         i = 0
        print(coch[0][:, :, 0].shape)
        inv_signal_l, inv_coch_l = invert_cochleagram(np.array(coch[0])[:, :, 0], 48000, 37, 30, 20000, 1, 0, n_iter=10, strict=False)
        inv_signal_r, inv_coch_r = invert_cochleagram(np.array(coch[0])[:, :, 1], 48000, 37, 30, 20000, 1, 0, n_iter=10, strict=False)
        inv_signal = np.stack([inv_signal_l, inv_signal_r], axis=1)
        slab.Binaural(inv_signal).play()
