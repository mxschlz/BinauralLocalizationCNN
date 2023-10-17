import random
import os
import tensorflow as tf


# Type conversion functions
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature_numpy(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


LBS_MAPPING = {
    'azim': ['train/azim', _int64_feature],
    'elev': ['train/elev', _int64_feature],
    'dist': ['train/dist', _float_feature],
    'hrtf_idx':  ['train/hrtf_idx', _int64_feature],
    'ITD':  ['train/ITD', _int64_feature],
    'ILD':  ['train/ILD', _int64_feature],
    'smooth_factor': ['train/smooth_factor', _int64_feature],
    'sampling_rate': ['train/sampling_rate', _int64_feature],
    'cnn_idx': ['train/cnn_idx', _int64_feature],
    'center_freq': ['train/center_freq', _int64_feature],
    'bandwidth': ['train/bandwidth', _float_feature],
    'n_sounds': ['train/n_sounds', _int64_feature]
}


# much easier to use if write a method to append a binaural sound and associated labels to the record
# cleaner version
def create_tfrecord_feature(subbands, labels=None):
    """
    prepare the data to be written into tf.record
    Args:
        subbands: binaural sound evoked activities after passing through cochleagram, see cochleagram_wrapper
        labels: dict, labels associated with the stimulus, depends on exp_type. could have following keys:
            azim: should be azim angle
            elev: elev angle
            freq: frequency
            modulation_delay: modulation delay
            flipped: flipped
            ITD:
            ILD:
            carrier_freq:
            modulation_freq:
            carrier_delay:
            modulation_delay:
            delay:
            start_sample:
            lead_level:
            lag_level:
            bandwidth:
            center_freq:
            noise_idx:
            subject_num:
            low_cutoff:
            high_cutoff:
            smooth_factor:
            speech:

    Returns:
        dict to be saved to tf.record
    """
    # every record should have those features
    feature = {'train/image': _bytes_feature(tf.compat.as_bytes(subbands.tobytes())),
               'train/image_height': _int64_feature(subbands.shape[0]),
               'train/image_width': _int64_feature(subbands.shape[1]),
               }

    # add information in the labels into feature dict
    if labels is not None:
        for k, v in labels.items():
            try:
                tf_label = LBS_MAPPING[k]
                feature[tf_label[0]] = tf_label[1](v)
            except KeyError:
                raise KeyError("feature: {} currently not implemented".format(k))

    return feature


def create_feature(subbands, labels=None):
    """
    prepare the data to be used directly for the CNN
    Args:
        subbands: binaural sound evoked activities after passing through cochleagram, see cochleagram_wrapper
        labels: dict, labels associated with the stimulus, depends on exp_type. could have following keys:

    Returns:
        dict to be saved to tf.record
    """
    # every record should have those features
    feature = {'train/image': _bytes_feature(tf.compat.as_bytes(subbands.tobytes())),
               'train/image_height': _int64_feature(subbands.shape[0]),
               'train/image_width': _int64_feature(subbands.shape[1]),
               }

    # add information in the labels into feature dict
    if labels is not None:
        for k, v in labels.items():
            try:
                tf_label = LBS_MAPPING[k]
                feature[tf_label[0]] = tf_label[1](v)
            except KeyError:
                raise KeyError("feature: {} currently not implemented".format(k))

    return feature


# TODO: tf does not support appending to a tfrecord
def create_tfrecord(stim_dicts, rec_path):
    """
    given a list of dicts containing binaural sounds and corresponding labels, create the tfrecord for the DNN model
    :param stim_dicts: output from CNN_preproc
    :param rec_path: name/path for the tfrecord to be created
    :return: None
    """
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(rec_path, options=options)

    # shuffle the list
    random.shuffle(stim_dicts)

    for stim_d in stim_dicts:
        subbands, lb = stim_d['subbands'], stim_d['label']

        rc = create_tfrecord_feature(subbands, lb)

        # write the single record into tfrecord file
        example = tf.train.Example(features=tf.train.Features(feature=rc))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()


def append_records_v1(in_file, new_records, out_file):
    """
    append a list of records to an existing tfrecord file
    Args:
        in_file: path, old file
        new_records: list/tuple of new records to be added
        out_file: path, new file

    Returns:
        None

    """
    # TODO: not sure if the same file name can be used
    assert not os.path.samefile(in_file, out_file), 'please give a new file name'
    with tf.io.TFRecordWriter(out_file) as writer:
        with tf.Graph().as_default(), tf.Session():
            ds = tf.data.TFRecordDataset([in_file])
            rec = ds.make_one_shot_iterator().get_next()
            while True:
                try:
                    writer.write(rec.eval())
                except tf.errors.OutOfRangeError:
                    break
        for new_rec in new_records:
            writer.write(new_rec)


def check_record(rec_file):
    """
    check if there is any error in created tfrecord file
    Args:
        rec_file:

    Returns:

    """
    reader_opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    record_iterator = tf.python_io.tf_record_iterator(path=rec_file,
                                                      options=reader_opts)
    # check for corrupted records
    rec_idx = 0
    try:
        for _ in record_iterator:
            rec_idx += 1
    except Exception as e:
        print('Error in {} at record {}'.format(rec_file, rec_idx))
        print(e)
        return False, rec_idx
    return True, rec_idx


# testing
if __name__ == '__main__':
    state, rec_idx = check_record(rec_file="/home/max/PycharmProjects/BinauralLocalizationCNN/numjudge_full_set_talkers_clear_train.tfrecords")
