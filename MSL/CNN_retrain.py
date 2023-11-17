import collections

from CNN_util import build_tfrecords_iterator, get_feature_dict, cost_function, freeze_session
from NetBuilder import NetBuilder

import os
import glob
import tensorflow as tf
import numpy as np
import time
import json
import pdb

# custom memory saving gradient
import memory_saving_gradients
from tensorflow.python.ops import gradients
from run_CNN import update_param_dict
from MSL.config_MSL import CONFIG_TRAIN as cfg
# import mem_util

# this controls CUDA convolution optimization
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'


# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(ys, xs, grad_ys, checkpoints='memory', **kwargs)


gradients.__dict__["gradients"] = memory_saving_gradients.gradients_speed


# data paths
stim_tfrec_pattern = "*train*.tfrecords"
stim_files = glob.glob(stim_tfrec_pattern)

# load config array from trained network
trainedNets_path = "netweights"
names = os.listdir(trainedNets_path)
net_dirs = sorted([n for n in names if os.path.isdir(os.path.join(trainedNets_path, n))])
curr_net = os.path.join(trainedNets_path, net_dirs[0])
config_fname = 'config_array.npy'
config_array = np.load(os.path.join(curr_net, config_fname), allow_pickle=True)

# default parameters

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

# from the stim_dset, create a batched data iterator
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

# build the CNN net
# load config array from trained network
net_name = os.path.split(curr_net)[-1]
config_fname = 'config_array.npy'
config_array = np.load(os.path.join(curr_net, config_fname), allow_pickle=True)

# network input
# the signal is power-transformed
# coch_sig = data_samp['train/image']
new_sig_nonlin = tf.pow(data_samp['train/image'], 0.3)

# build the neural network
net = NetBuilder(cpu_only=cfg["DEFAULT_NET_PARAM"]['cpu_only'])
net_out = net.build(config_array, new_sig_nonlin, **net_params)

# regularizer
if net_params['regularizer'] is not None:
    reg_term = tf.contrib.layers. \
        apply_regularization(net_params['regularizer'],
                             (tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

# get network cost and labels
cost, net_labels, multihot_labels = cost_function(data_samp, net_out, **cost_params)
if net_params['regularizer'] is not None:
    cost = tf.add(cost, reg_term)

# network outputs
# network maximum-likelihood prediction
"""
cond_dist = tf.nn.softmax(net_out)
net_pred = tf.argmax(cond_dist, 1)
top_k = tf.nn.top_k(net_out, 5)
# correct predictions
correct_pred = tf.equal(tf.argmax(net_out, 1), tf.cast(net_labels, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
"""

cond_dist = tf.nn.sigmoid(net_out)
auc, update_op_auc = tf.metrics.auc(multihot_labels, cond_dist)
# Evaluate model
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
testing = run_params["testing"]


# search for dense layer weights or posterior
def unique_list(list1, list2):
    return [x for x in list1 if x not in list2]


# all variables
all_vars = list(set(v.op.name for v in tf.global_variables()).difference([]))

# retrain variables
retrain_vars = list()
for weight in all_vars:
    if weight.find("fc") != -1:
        retrain_vars.append(weight)
    if weight.find("out") != -1:
        retrain_vars.append(weight)

# get freeze variables for debugging
freeze_vars = unique_list(all_vars, retrain_vars)

# get variable list for restoring in saver
var_list = tf.contrib.framework.get_variables_to_restore(exclude=None)


if testing:
    print("Please set testing param to False in order to retrain the CNN!")
if not testing:
    # model_weights = os.path.join(curr_net, "model.ckpt-" + model_version[0])
    # ckpt = tf.train.load_checkpoint(model_weights)
    newpath = trainedNets_path + "_MSL/" + net_name
    display_step = run_params["display_step"]
    sess.run(stim_iter.initializer)
    saver = tf.train.Saver(max_to_keep=None, var_list=var_list)
    learning_curve = []
    errors_count = 0
    step = 1
    try:
        # sess.graph.finalize()
        # sess.run(partially_frozen)
        while True:
            # sess.run([optimizer,check_op])
            try:
                if step == 1:
                    # saver.restore(sess, model_weights)
                    freeze_session(sess, keep_var_names=retrain_vars)  # freeze all layers prior to dense layer
                    sess.run(update_grads)
                else:
                    sess.run(update_grads)
            # sess.run(update_grads)
            except tf.errors.InvalidArgumentError as e:
                print(e.message)
                errors_count += 1
                continue
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc, idx, auc_out = sess.run([cost, accuracy, data_label['train/cnn_idx'], auc, update_op_auc])
                # print("Batch Labels: ",az)
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc) + ", AUC= " + "{:.5f}".format(auc))
            if step % run_params["checkpoint_step"] == 0:
                print("Checkpointing Model...")
                retry_count = 0
                while True:
                    try:
                        saver.save(sess, newpath + f'/model.ckpt', global_step=step,
                                   write_meta_graph=False)
                        break
                    except ValueError as e:
                        if retry_count > 36:
                            print("Maximum wait time reached(6H). Terminating Program.")
                            raise e from None
                        print("Checkpointing failed. Retrying in 10 minutes...")
                        time.sleep(600)
                        retry_count += 1
                learning_curve.append([int(step * batch_size), float(acc)])
                print("Checkpoint Complete")

            # Just for testing the model/call_model
            if step == run_params["total_steps"]:
                print("Break!")
                break
            step += 1
            print(f"Current step: {step}")
    except tf.errors.OutOfRangeError:
        print("Out of Range Error. Optimization Finished")
    except tf.errors.DataLossError as e:
        print("Corrupted file found!!")
        pdb.set_trace()
    finally:
        print("Total errors: ", errors_count)
        print("Training stopped.")

    with open(newpath + '/learning_curve_retrained.json', 'w') as f:
        json.dump(learning_curve, f)

# cleanup
sess.close()
tf.reset_default_graph()
