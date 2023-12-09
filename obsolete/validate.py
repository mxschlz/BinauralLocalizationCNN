from CNN_util import build_tfrecords_iterator, get_feature_dict, cost_function
from NetBuilder import NetBuilder

import os
import glob
import tensorflow as tf
import numpy as np
import collections
import csv
from copy import deepcopy
import time
# from CNN_util import freeze_session
import pdb
import json
from analysis_and_plotting.decision_rule import decide_sound_presence
from augment import apply_random_augmentation
from run_CNN import update_param_dict

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


cfg = dict(DEFAULT_NET_PARAM={'cpu_only': False, 'regularizer': None, "n_classes_localization": 504},
           DEFAULT_COST_PARAM={"multi_source_localization": True},
           DEFAULT_RUN_PARAM={'learning_rate': 1e-3,
                              'batch_size': 16,
                              'model_version': [str(x) for x in range(250, 10000, 250)]},
           DEFAULT_DATA_PARAM={"augment": True}
           )

# data paths
stim_tfrec_pattern = "*test*.tfrecords"
stim_files = glob.glob(stim_tfrec_pattern)

# load config array from trained network
trainedNets_path = "netweights_MSL"
names = os.listdir(trainedNets_path)
net_dirs = sorted([n for n in names if os.path.isdir(os.path.join(trainedNets_path, n))])
curr_net = os.path.join(trainedNets_path, net_dirs[0])
config_fname = 'config_array.npy'
config_array = np.load(os.path.join(curr_net, config_fname), allow_pickle=True)

# additional params
ds_params = {}
net_params = {}
cost_params = {}
run_params = {}

ds_params = update_param_dict(cfg["DEFAULT_DATA_PARAM"], ds_params)
net_params = update_param_dict(cfg["DEFAULT_NET_PARAM"], net_params)
run_params = update_param_dict(cfg["DEFAULT_RUN_PARAM"], run_params)
cost_params = update_param_dict(cfg["DEFAULT_COST_PARAM"], cost_params)

# build dataset iterator
stim_files = glob.glob(stim_tfrec_pattern)
stim_feature = get_feature_dict(stim_files[0])
stim_dset = build_tfrecords_iterator(stim_tfrec_pattern, stim_feature, **ds_params)

batch_size = cfg["DEFAULT_RUN_PARAM"]['batch_size']
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
if "augment" in ds_params:
    data_samp["train/image"] = apply_random_augmentation(data_samp["train/image"])

# build the CNN net
# load config array from trained network
net_name = os.path.split(curr_net)[0]
config_fname = 'config_array.npy'
config_array = np.load(os.path.join(curr_net, config_fname), allow_pickle=True)

# network input
# the signal is power-transformed
# coch_sig = data_samp['train/image']
new_sig_nonlin = tf.pow(data_samp['train/image'], 0.3)

# build the neural network
net = NetBuilder(cpu_only=cfg["DEFAULT_NET_PARAM"]['cpu_only'])
net_out = net.build(config_array, new_sig_nonlin, **net_params)

# get network cost and labels
cost, net_labels = cost_function(data_samp, net_out, **cost_params)

# network outputs
cond_dist = tf.nn.sigmoid(net_out)
auc, update_op_auc = tf.metrics.auc(net_labels, cond_dist, name='auc')
running_vars_auc = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                     scope='auc')
running_vars_auc_initializer = tf.variables_initializer(var_list=running_vars_auc)
# Evaluate model
correct_pred = tf.equal(tf.to_int64(net_out > 0.5), tf.cast(net_labels, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

# TODO: parallelization options decide how much memory is needed. 0 is too many.
#  for testing use 1, but should try to see how many is optimal
config = tf.ConfigProto(allow_soft_placement=True,
                        inter_op_parallelism_threads=0,
                        intra_op_parallelism_threads=0)
sess = tf.Session(config=config)
sess.run(init_op)

# file handle for .csv writing
data_label = dict(sorted(data_label.items()))  # sort dict so that the order stays the same over all model results
eval_keys = list(data_label.keys())
eval_vars = list(data_label.values())

# only consider testing
model_version = run_params['model_version']

# model_weights = os.path.join(curr_net, "model.ckpt-" + model_version[0])
# ckpt = tf.train.load_checkpoint(model_weights)
sess.run(stim_iter.initializer)
auc_results = dict()
for mv_num in model_version:
    auc_results[mv_num] = []
    print("Starting model version: ", mv_num)
    saver = tf.train.Saver(max_to_keep=None)
    saver.restore(sess, os.path.join(curr_net, "model.ckpt-" + f"{mv_num}"))
    while True:
        try:
            # Calculate batch loss and accuracy
            loss, acc, auc_out, update_auc_out = sess.run([cost, accuracy, auc, update_op_auc])
            auc_results[mv_num].append(update_auc_out)
            print(f"AUC = {update_auc_out}")
        except tf.errors.OutOfRangeError:
            print('Dataset finished')
            break
        finally:
            pass

# cleanup
sess.close()
tf.reset_default_graph()
