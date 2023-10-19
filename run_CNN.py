from CNN_util import build_tfrecords_iterator, get_feature_dict, cost_function
from NetBuilder import NetBuilder

import os
import glob
import tensorflow as tf
import numpy as np
import collections
import csv
from copy import deepcopy


# custom memory saving gradient
import memory_saving_gradients
from tensorflow.python.ops import gradients
# import mem_util

# this controls CUDA convolution optimization
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'


# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(ys, xs, grad_ys, checkpoints='memory', **kwargs)


gradients.__dict__["gradients"] = memory_saving_gradients.gradients_speed


def update_param_dict(default_dict, usr_dict):
    """
    update the default dict with key/values in usr_dict
    :param default_dict:
    :param usr_dict:
    :return: a new dict
    """
    res_dict = deepcopy(default_dict)
    for k, v in usr_dict.items():
        default_dict[k] = v
    return res_dict


def run_CNN(stim_tfrec_pattern, trainedNet_path, save_name, cfg,
            ds_params={}, net_params={}, cost_params={}, run_params={}):
    """
    run the CNN model from Francl 2022
    :param stim_tfrec_pattern: a regexpr pattern to read all the input tfrecords
    :param trainedNet_path: path points to the folder of the trained CNN model
    :param save_name: file name prefix used to save the result file
    :param ds_params: parameters for dataset loading, see build_tfrecords_iterator
    :param net_params: parameters for net building, see NetBuilder
    :param cost_params: parameters for the cost function, see cost_function
    :param run_params: parameters for running the model
    :return:
    """
    # input checking
    ds_params = update_param_dict(cfg["DEFAULT_DATA_PARAM"], ds_params)
    net_params = update_param_dict(cfg["DEFAULT_NET_PARAM"], net_params)
    run_params = update_param_dict(cfg["DEFAULT_RUN_PARAM"], run_params)
    cost_params = update_param_dict(cfg["DEFAULT_COST_PARAM"], cost_params)

    # build dataset iterator
    stim_files = glob.glob(stim_tfrec_pattern)
    stim_feature = get_feature_dict(stim_files[0])
    stim_dset = build_tfrecords_iterator(stim_tfrec_pattern, stim_feature, **ds_params)

    # from the stim_dset, create a batched data iterator
    batch_size = run_params['batch_size']
    batch_size_tf = tf.constant(batch_size, dtype=tf.int64)
    stim_dset = stim_dset.shuffle(buffer_size=batch_size). \
        batch(batch_size=batch_size_tf, drop_remainder=True)
    stim_iter = stim_dset.make_initializable_iterator()
    data_samp = stim_iter.get_next()
    # get a data label dict from the data sample
    data_label = collections.OrderedDict()
    for k, v in data_samp.items():
        if k not in ('train/image', 'train/image_height', 'train/image_width'):
            data_label[k] = data_samp[k]

    # build the CNN net
    # load config array from trained network
    net_name = os.path.split(trainedNet_path)[-1]
    config_fname = 'config_array.npy'
    config_array = np.load(os.path.join(trainedNet_path, config_fname), allow_pickle=True)

    # network input
    # the signal is power-transformed
    # coch_sig = data_samp['train/image']
    new_sig_nonlin = tf.pow(data_samp['train/image'], 0.3)

    # build the neural network
    net = NetBuilder(cpu_only=net_params['cpu_only'])
    net_out = net.build(config_array, new_sig_nonlin, **net_params)

    # regularizer
    if net_params['regularizer'] is not None:
        reg_term = tf.contrib.layers. \
            apply_regularization(net_params['regularizer'],
                                 (tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

    # get network cost and labels
    cost, net_labels = cost_function(data_samp, net_out, **cost_params)
    if net_params['regularizer'] is not None:
        cost = tf.add(cost, reg_term)

    # network outputs
    # network maximum-likelihood prediction
    cond_dist = tf.nn.softmax(net_out)
    net_pred = tf.argmax(cond_dist, 1)
    top_k = tf.nn.top_k(net_out, 5)
    # correct predictions
    correct_pred = tf.equal(tf.argmax(net_out, 1), tf.cast(net_labels, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # launch the model
    is_testing = run_params['testing']
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        update_grads = tf.train.AdamOptimizer(learning_rate=run_params['learning_rate'],
                                              epsilon=1e-4).minimize(cost)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # TODO: parallelization options decide how much memory is needed. 0 is too many.
    #  for testing use 1, but should try to see how many is optimal
    config = tf.ConfigProto(allow_soft_placement=True,
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    sess = tf.Session(config=config)
    sess.run(init_op)

    # file handle for .csv writing
    data_label = dict(sorted(data_label.items()))  # sort dict so that the order stays the same over all model results
    eval_keys = list(data_label.keys())
    eval_vars = list(data_label.values())

    # only consider testing
    model_version = run_params['model_version']
    for mv_num in model_version:
        sess.run(stim_iter.initializer)
        # load model
        print("Starting model version: ", mv_num)
        saver = tf.train.Saver(max_to_keep=None)
        saver.restore(sess, os.path.join(trainedNet_path, "model.ckpt-190"))

        header = ['model_pred'] + eval_keys
        # header = ['model_pred'] + eval_keys + ['cnn_idx_' + str(i) for i in range(504)]
        csv_path = "{}_model_{}_{}.csv".format(save_name, net_name, mv_num)
        csv_handle = open(csv_path, 'w', encoding='UTF8', newline='')
        csv_writer = csv.writer(csv_handle)
        csv_writer.writerow(header)

        while True:
            # running individual batches
            try:
                pd, pd_corr, cd, e_vars = sess.run([net_pred, correct_pred, cond_dist, eval_vars])
                # prepare result to write into .csv
                csv_rows = list(zip(pd, *e_vars))
                # csv_rows = list(zip(pd, *e_vars, cd.tolist()))
                csv_writer.writerows(csv_rows)
            except tf.errors.ResourceExhaustedError:
                print("Out of memory error")
                break
            except tf.errors.OutOfRangeError:
                print('Dataset finished')
                break

            finally:
                pass

        # close the csv file
        csv_handle.close()

    # cleanup
    sess.close()
    tf.reset_default_graph()


if __name__ == '__main__':
    netweights_path = "/home/max/PycharmProjects/BinauralLocalizationCNN/netweights_MSL/"
    first_net_path = os.path.join(netweights_path, sorted(os.listdir(netweights_path))[0])
    config_fname = 'config_array.npy'
    config_array = np.load(os.path.join(first_net_path, config_fname), allow_pickle=True)
    stim_tfrecs = os.path.join('tfrecords', 'msl', "numjudge_*test.tfrecords")
    res_name = os.path.join('Result', 'NumJudge_result')
    run_CNN(stim_tfrecs, first_net_path, res_name)

