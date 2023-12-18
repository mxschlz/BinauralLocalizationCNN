import glob
from CNN_util import get_feature_dict, build_tfrecords_iterator
import tensorflow as tf


# training data path
train_data_path = "tfrecords/msl/*train.tfrecords"
# get all files
stim_files = glob.glob(train_data_path)
# get feature dict
stim_feature = get_feature_dict(stim_files[0])
# get the stim dataset
stim_dset = build_tfrecords_iterator(train_data_path, stim_feature)
stim_iter = stim_dset.make_initializable_iterator()
data_samp = stim_iter.get_next()
# augment new data from dataset
stim_dset1 = stim_dset.map(lambda image: (tf.image.random_flip_left_right(image, 15)))
