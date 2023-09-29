import collections
import csv

from CNN_util import build_tfrecords_iterator, get_feature_dict, cost_function
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
# import mem_util

# this controls CUDA convolution optimization
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'


# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(ys, xs, grad_ys, checkpoints='memory', **kwargs)


gradients.__dict__["gradients"] = memory_saving_gradients.gradients_speed


# data paths
stim_tfrec_pattern = "tfrecords/msl/locaaccu_noise_v.tfrecords"
stim_files = glob.glob(stim_tfrec_pattern)
save_name = os.path.join('Result', 'locaaccu_noise_v')

# load config array from trained network
trainedNets_path = "netweights"
names = os.listdir(trainedNets_path)
net_dirs = sorted([n for n in names if os.path.isdir(os.path.join(trainedNets_path, n))])
curr_net = os.path.join(trainedNets_path, net_dirs[0])
config_fname = 'config_array.npy'
config_array = np.load(os.path.join(curr_net, config_fname), allow_pickle=True)

# default parameters
DEFAULT_DATA_PARAM = {}
DEFAULT_NET_PARAM = {'cpu_only': True, 'regularizer': None}
DEFAULT_COST_PARAM = {}
DEFAULT_RUN_PARAM = {'learning_rate': 1e-3,
                     'batch_size': 16,
                     'testing': True,
                     'model_version': ['100000']}

# additional params
ds_params = {}
net_params = {}
cost_params = {}
run_params = {}

ds_params = update_param_dict(DEFAULT_DATA_PARAM, ds_params)
net_params = update_param_dict(DEFAULT_NET_PARAM, net_params)
run_params = update_param_dict(DEFAULT_RUN_PARAM, run_params)
cost_params = update_param_dict(DEFAULT_COST_PARAM, cost_params)

# build dataset iterator
stim_files = glob.glob(stim_tfrec_pattern)
stim_feature = get_feature_dict(stim_files[0])
stim_dset = build_tfrecords_iterator(stim_tfrec_pattern, stim_feature, **ds_params)

# from the stim_dset, create a batched data iterator
batch_size = DEFAULT_RUN_PARAM['batch_size']
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
net = NetBuilder(cpu_only=DEFAULT_NET_PARAM['cpu_only'])
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
testing = run_params["testing"]
for mv_num in model_version:
    if testing:
        sess.run(stim_iter.initializer)
        # load model
        print("Starting model version: ", mv_num)
        saver = tf.train.Saver(max_to_keep=None)
        saver.restore(sess, os.path.join(curr_net, "model.ckpt-" + str(mv_num)))

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

    if not testing:
        newpath = trainedNets_path + "_retrained"
        num_files = len(stim_files)
        display_step = 25
        sess.run(stim_iter.initializer)
        saver = tf.train.Saver(max_to_keep=None)
        learning_curve = []
        errors_count = 0
        try:
            step = 1
            sess.graph.finalize()
            while True:
                # sess.run([optimizer,check_op])
                try:
                    if step == 1:
                        if not num_files == 1:
                            # latest_addition = max(files, key=os.path.getctime)
                            file_list = []
                            for elem in stim_files:
                                if (elem.split("/")[-1]).split(".")[0] == 'model':
                                    file_list.append(elem)
                            latest_addition = max(file_list, key=os.path.getctime)
                            latest_addition_name = latest_addition.split(".")[-2]
                            saver.restore(sess, newpath + "/model." + latest_addition_name)
                            step = int(latest_addition_name.split("-")[1])
                        else:
                            sess.run(update_grads)
                    else:
                        sess.run(update_grads)
                #                    sess.run(update_grads)
                except tf.errors.InvalidArgumentError as e:
                    print(e.message)
                    errors_count += 1
                    continue
                if step % display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc, n_sources = sess.run([cost, accuracy, data_label[0]['train/n_sources']])
                    # print("Batch Labels: ",az)
                    print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))
                if step % 5000 == 0:
                    print("Checkpointing Model...")
                    retry_count = 0
                    while True:
                        try:
                            saver.save(sess, newpath + '/model.ckpt', global_step=step, write_meta_graph=False)
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
                if step == 300000:  # TODO: was 300000 but set it to 10000 according to CNN paper MSL retraining
                    print("Break!")
                    break
                step += 1
        except tf.errors.OutOfRangeError:
            print("Out of Range Error. Optimization Finished")
        except tf.errors.DataLossError as e:
            print("Corrupted file found!!")
            pdb.set_trace()
        finally:
            print(errors_count)
            print("Training stopped.")

        with open(newpath + '/curve_no_resample_w_cutoff_vary_loc.json', 'w') as f:
            json.dump(learning_curve, f)

    # cleanup
    sess.close()
    tf.reset_default_graph()
