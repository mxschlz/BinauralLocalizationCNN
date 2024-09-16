"""
utilities to run the CNN model from Francl 2022 paper
"""
import collections
import glob
import json

import numpy as np
import tensorflow as tf
from google.protobuf.json_format import MessageToJson

# TODO: remapping of azim/elev data; keep original, but seems azim in [0, 360] and elev in [0, 60]


DATA_MAPPING = {
    'azim': ['train/azim', tf.int64],
    'elev': ['train/elev', tf.int64],
    'dist': ['train/dist', tf.float32],
    "n_sounds": ["train/n_sounds", tf.int64],
    'hrtf_idx': ['train/hrtf_idx', tf.int64],
    'sampling_rate': ['train/sampling_rate', tf.int64],
    'ITD': ['train/ITD', tf.int64],
    'ILD': ['train/ILD', tf.int64],
    'smooth_factor': ['train/smooth_factor', tf.int64],
    'center_freq': ['train/center_freq', tf.int64],
    'bandwidth': ['train/bandwidth', tf.float32],
}


def remap_stim_dict(stim_dict):
    """
    re-map the individual dictionary for each stimulus as a tensor dict which can be used for the CNN
    :param stim_dict: single dict, output from CNN_preproc
    :return:
    """
    subbands = stim_dict['subbands']
    feature = {'train/image': tf.convert_to_tensor(subbands, dtype=tf.float32),
               'train/image_height': tf.convert_to_tensor(subbands.shape[0], dtype=tf.int64),
               'train/image_width': tf.convert_to_tensor(subbands.shape[1], dtype=tf.int64),
               }
    d_types = [tf.float32, tf.int64, tf.int64]

    labels = stim_dict['label']
    # add information in the labels into feature dict
    if labels is not None:
        for k, v in labels.items():
            try:
                tf_label = DATA_MAPPING[k]
                feature[tf_label[0]] = tf.convert_to_tensor(v, dtype=tf_label[1])
                d_types.append(tf_label[1])
            except KeyError:
                raise KeyError("feature: {} currently not implemented".format(k))

    return feature


class DatasetFromList:

    def __init__(self, stim_dicts):
        self.data = stim_dicts

    def __call__(self, *args, **kwargs):
        for d in self.data:
            res, _ = remap_stim_dict(d)
            yield res


def build_tfrecords_iterator(train_path_pattern, feature_parsing_dict,
                             is_bkgd=False, narrowband_noise=False,
                             manually_added=False, localization_bin_resolution=5,
                             stacked_channel=True, ds_ratio=6, **kwargs):
    """
    Builds tensorflow iterator for feeding graph with data from tfrecords.

    :param train_path_pattern: regexpr pattern to locate the tfrecords files
    :param feature_parsing_dict: tfrecords feature dict, see get_feature_dict
    :param is_bkgd: bool
    :param narrowband_noise: bool
    :param manually_added: bool
    :param localization_bin_resolution: int, degree
    :param stacked_channel: bool
    :param ds_ratio: int, down-sample ration for the data
    :return: tf.data.Dataset iterator
    """
    if stacked_channel:
        STIM_SIZE = [39, int(48000 / ds_ratio), 2]
    else:
        STIM_SIZE = [78, int(48000 / ds_ratio)]

    # get all the tfrecords files to be used
    training_paths = glob.glob(train_path_pattern)

    # check if is original data or custom-made data from current tool
    is_origin = True
    if 'train/cnn_idx' in feature_parsing_dict:
        is_origin = False

    # Set up feature_dict to use for parsing tfrecords
    feature_dict = collections.OrderedDict()
    for key in feature_parsing_dict.keys():
        if is_bkgd is True and (key == 'train/azim' or key == 'train/elev'):
            val_dtype = feature_parsing_dict[key].dtype
            feature_dict[key] = tf.VarLenFeature(val_dtype)
        else:
            val_dtype = feature_parsing_dict[key].dtype
            val_shape = feature_parsing_dict[key].shape
            feature_dict[key] = tf.FixedLenFeature(val_shape, val_dtype)

    # Define the tfrecords parsing function
    def parse_single_tfrecord(record):
        """
        parse a single tfrecord entry
        Parsing function returns dictionary of tensors with tfrecords paths as keys
        :param record:
        :return:
        """
        # Parse the record read by the reader
        parsed_features = tf.parse_single_example(record, features=feature_dict)
        input_tensor_dict = {}
        # for path in sorted(feature_parsing_dict.keys()):
        for key in feature_parsing_dict.keys():
            if is_bkgd is True and (key == 'train/azim' or key == 'train/elev'):
                val_dtype = feature_parsing_dict[key].dtype
                input_tensor_dict[key] = parsed_features[key]
            else:
                val_dtype = feature_parsing_dict[key].dtype
                val_shape = feature_parsing_dict[key].shape
                if key == 'train/image':
                    # if len(path_shape) > 0: # Array-like features are read-in as bytes and must be decoded
                    if val_dtype == tf.string:
                        val_dtype = tf.float32
                    decoded_bytes_feature = tf.decode_raw(parsed_features[key], val_dtype)
                    if decoded_bytes_feature.dtype == tf.float64:
                        # This will cast tf.float64 inputs to tf.float32, since many tf ops do not support tf.float64.
                        # If we want control over this (i.e. make the network run using tf.float16, we should either
                        # change the tfrecords files or add a cast operation after calling the iterator).
                        decoded_bytes_feature = tf.cast(decoded_bytes_feature, tf.float32)
                    input_tensor_dict[key] = tf.reshape(decoded_bytes_feature, STIM_SIZE)
                else:
                    input_tensor_dict[key] = parsed_features[key]

        # type casting; not sure why is needed anyway
        label_div_const = tf.constant([localization_bin_resolution])
        for elem in input_tensor_dict:
            if elem != 'train/image':
                if input_tensor_dict[elem].dtype == 'float32':
                    v = tf.cast(input_tensor_dict[elem], tf.float32)
                    input_tensor_dict[elem] = v
                elif input_tensor_dict[elem].dtype == 'int64':
                    v = tf.cast(input_tensor_dict[elem], tf.int32)
                    if is_origin:
                        if elem in ('train/azim', 'train/elev'):
                            # convert angle value to bin index
                            if not is_bkgd and (narrowband_noise or not manually_added):
                                v = tf.div(v, label_div_const)
                    input_tensor_dict[elem] = v

        images = tf.slice(input_tensor_dict['train/image'], [0] * len(STIM_SIZE), STIM_SIZE)
        input_tensor_dict['train/image'] = images
        return input_tensor_dict

    dataset = tf.data.Dataset.list_files(train_path_pattern).shuffle(len(training_paths))
    cl = min(len(training_paths), 10)
    dataset = dataset.apply(tf.contrib.data.parallel_interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP").
        map(parse_single_tfrecord, num_parallel_calls=1),
        cycle_length=cl, block_length=16)
    )
    # TODO: check buffer_size, currently seems 200 is too large for my computer
    dataset = dataset.shuffle(buffer_size=32)
    dataset = dataset.prefetch(16)

    return dataset


DEFAULT_COMP_OPT = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)


def get_feature_dict(tf_file, tf_opts=DEFAULT_COMP_OPT, is_bkgd=False):
    """
    get the feature dict needed to construct dataset iterator from tfrecords
    :param tf_file: a single sample tfrecords file
    :param tf_opts: compression used in the tfrecords file
    :param is_bkgd: bool, if the tfrecords is about background sound
    :return:
    """
    # read a single record and decode it using json
    tfr_iter = tf.python_io.tf_record_iterator(tf_file, options=tf_opts)
    tfr_samp = next(tfr_iter)
    samp_js = MessageToJson(tf.train.Example.FromString(tfr_samp))
    jsdict = json.loads(samp_js)
    # close the iterator
    tfr_iter.close()

    feature = collections.OrderedDict()
    for k, v in sorted(jsdict.items()):
        v = v['feature']
        key1 = v.keys()
        for x in key1:
            key2 = v[x].keys()
            for y in key2:
                if y == 'int64List':
                    val_dtype = tf.int64
                elif y == 'bytesList':
                    val_dtype = tf.string
                elif y == 'floatList':
                    val_dtype = tf.float32
                else:
                    raise KeyError("conversion of data type {} not implemented".format(y))
                feature_len = len(v[x][y]['value'])
                shape = [] if feature_len == 1 else [feature_len]
                if is_bkgd is True and (x == 'train/azim' or x == 'train/elev'):
                    feature[x] = tf.VarLenFeature(val_dtype)
                else:
                    feature[x] = tf.FixedLenFeature(shape, val_dtype)

    return feature


def cost_function(data_sample, net_out, sam_tones=False, transposed_tones=False,
                  precedence_effect=False, tone_version=False, multi_source_localization=False):
    """
    cost function for the CNN
    :param data_sample: data returned from dataset iterator
    :param net_out: output from the CNN
    :param sam_tones:
    :param transposed_tones:
    :param precedence_effect:
    :param tone_version:
    :param multi_source_localization:
    :return:
    """
    is_origin = True
    if 'train/cnn_idx' in data_sample:
        is_origin = False

    if sam_tones or transposed_tones:
        labels_batch_cost_sphere = tf.squeeze(tf.zeros_like(data_sample['train/carrier_freq']))
    elif precedence_effect:
        labels_batch_cost_sphere = tf.squeeze(tf.zeros_like(data_sample['train/start_sample']))
    elif multi_source_localization:
        labels_batch_sphere = data_sample['train/binary_label']
        if not isinstance(labels_batch_sphere, tf.SparseTensor):
            labels_batch_sphere = tf.sparse.from_dense(labels_batch_sphere)
            # labels_batch_cost_sphere = tf.squeeze(tf.zeros_like(data_sample["train/cnn_idxs"]))
        multihot_labels = tf.cast(tf.sparse.to_indicator(labels_batch_sphere, 504),
                                  tf.float32)
    else:
        if not tone_version:
            # pos to label conversion, only for original data
            if is_origin:
                labels_batch_sphere = tf.add(tf.scalar_mul(tf.constant(36, dtype=tf.int32),
                                                           data_sample['train/elev']),
                                             data_sample['train/azim'])
            else:
                labels_batch_sphere = data_sample['train/cnn_idx']
        else:
            labels_batch_sphere = data_sample['train/azim']
        labels_batch_cost_sphere = tf.squeeze(labels_batch_sphere)
    if multi_source_localization:
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net_out,
                                                                      labels=multihot_labels))  # check here. Logits is model estimate, labels are true labels. Think i got the right labels
        return cost, multihot_labels
    else:
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                              (logits=net_out, labels=labels_batch_cost_sphere))
        return cost, labels_batch_cost_sphere


def get_dataset_partitions(ds, train_split=0.8, test_split=0.2, shuffle=True):
    """
    Splits a dataset into training, testing, and validation partitions.

    Parameters:
    - ds (list): The dataset to be partitioned.
    - train_split (float): The proportion of the dataset to allocate for training (default is 0.8).
    - test_split (float): The proportion of the dataset to allocate for testing (default is 0.1).
    - val_split (float): The proportion of the dataset to allocate for validation (default is 0.1).
    - shuffle (bool): Whether to shuffle the dataset before partitioning (default is True).

    Returns:
    - tuple: A tuple containing three lists - training dataset, testing dataset, and validation dataset.
    """

    # Ensure that the sum of splits is equal to 1
    assert (train_split + test_split) == 1

    # Shuffle the dataset if specified
    if shuffle:
        np.random.shuffle(ds)  # shuffles in-place

    # Calculate the sizes of each partition
    ds_size = len(ds)
    train_size = int(train_split * ds_size)
    test_size = int(test_split * ds_size)

    # Partition the dataset
    train_ds = ds[:train_size]
    test_ds = ds[train_size:train_size + test_size]

    return train_ds, test_ds


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def filter_sparse_to_dense(var_list):
    '''
    Finds Sparse tensors in list and converts to dense tensors. Returns list
    where sparse tenors are wrapped with a tf.sparse.to_dense operation so that
    any evaluation of the variables in the graph will only retun normal
    Tensors. Useful for writing out values to numpy arrays.

    Parameters:
        var_list (list) : List of unevaluated tensors
    Return:
        ret_list (list) : Same list of tensors with SparseTensors wrapped with
        a tf.sparse.to_dense operation so evaluation will result in a standard
        tensor.
    '''
    ret_list = []
    for var in var_list:
        if isinstance(var, tf.SparseTensor):
            var = tf.sparse.to_dense(var, default_value=-1)
        ret_list.append(var)
    return ret_list


def main() -> None:
    import os
    import NetBuilder

    file_pattern = '*train*.tfrecords'
    files = glob.glob(file_pattern)
    rec_feature = get_feature_dict(files[0])
    # build the dataset iterator
    dataset = build_tfrecords_iterator(file_pattern, rec_feature)
    # from the stim_dset, create a batched data iterator
    batch_size = 16
    batch_size_tf = tf.constant(batch_size, dtype=tf.int64)
    stim_dset = dataset.shuffle(buffer_size=batch_size). \
        batch(batch_size=batch_size_tf, drop_remainder=True)
    stim_iter = stim_dset.make_initializable_iterator()
    data_sample = stim_iter.get_next()

    trainedNets_path = "netweights"
    names = os.listdir(trainedNets_path)
    net_dirs = sorted([n for n in names if os.path.isdir(os.path.join(trainedNets_path, n))])
    curr_net = os.path.join(trainedNets_path, net_dirs[0])
    net_name = os.path.split(curr_net)[-1]
    config_fname = 'config_array.npy'
    config_array = np.load(os.path.join(curr_net, config_fname), allow_pickle=True)

    # network input
    # the signal is power-transformed
    # coch_sig = data_samp['train/image']
    new_sig_nonlin = tf.pow(data_sample['train/image'], 0.3)

    # build the neural network
    net = NetBuilder.NetBuilder(cpu_only=True)
    net_out = net.build(config_array, new_sig_nonlin, regularizer=None)
    cost, labels = cost_function(data_sample, net_out, multi_source_localization=True)


if __name__ == "__main__":
    main()
