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


def run_CNN(stim_tfrec_pattern, trainedNet_path, cfg, save_name=None,
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
    is_msl = cfg["DEFAULT_COST_PARAM"]["multi_source_localization"]
    # from the stim_dset, create a batched data iterator
    batch_size = run_params['batch_size']
    batch_size_tf = tf.constant(batch_size, dtype=tf.int64)
    stim_dset = stim_dset.shuffle(buffer_size=batch_size).batch(batch_size=batch_size_tf, drop_remainder=True)
    stim_iter = stim_dset.make_initializable_iterator()
    data_samp = stim_iter.get_next()
    if "augment" in ds_params:
        data_samp["train/image"] = apply_random_augmentation(data_samp["train/image"])
    # get a data label dict from the data sample
    data_label = collections.OrderedDict()
    for k, v in data_samp.items():
        if k not in ('train/image', 'train/image_height', 'train/image_width'):
            data_label[k] = data_samp[k]

    # build the CNN net
    # load config array from trained network
    net_weights, net_name = os.path.split(trainedNet_path)
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
    if is_msl:  # multi source localization
        cost, net_labels = cost_function(data_samp, net_out, **cost_params)
        cond_dist = tf.nn.sigmoid(net_out)
        auc, update_op_auc = tf.metrics.auc(net_labels, cond_dist, name='auc')
        running_vars_auc = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                             scope='auc')
        running_vars_auc_initializer = tf.variables_initializer(var_list=running_vars_auc)
        # Evaluate model
        correct_pred = tf.equal(tf.to_int64(net_out > 0.5), tf.cast(net_labels, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    elif not is_msl:  # single sound source localization
        cost, net_labels = cost_function(data_samp, net_out, **cost_params)
        cond_dist = tf.nn.softmax(net_out)
        # Evaluate model
        net_pred = tf.argmax(cond_dist, 1)
        top_k = tf.nn.top_k(net_out, 5)
        # correct predictions
        correct_pred = tf.equal(tf.argmax(net_out, 1), tf.cast(net_labels, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    if net_params['regularizer'] is not None:
        cost = tf.add(cost, reg_term)

    # launch the model
    first_fc_idx = [x.name for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)].index('wc_fc_0:0')
    late_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[first_fc_idx:]
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if is_msl:
            update_grads = (tf.train.AdamOptimizer(learning_rate=run_params['learning_rate'],epsilon=1e-4)
                            .minimize(cost,var_list=late_layers))
        else:
            update_grads = tf.train.AdamOptimizer(learning_rate=run_params['learning_rate'],
                                                  epsilon=1e-4).minimize(cost)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # TODO: parallelization options decide how much memory is needed. 0 is too many.
    #  for testing use 1, but should try to see how many is optimal
    config = tf.ConfigProto(allow_soft_placement=True,
                            inter_op_parallelism_threads=0,
                            intra_op_parallelism_threads=0)
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init_op)

    # file handle for .csv writing
    data_label = dict(sorted(data_label.items()))  # sort dict so that the order stays the same over all model results
    eval_keys = list(data_label.keys())
    eval_vars = list(data_label.values())

    testing = run_params["testing"]
    model_version = run_params['model_version']
    if testing:
        for mv_num in model_version:
            sess.run(stim_iter.initializer)
            # load model
            print("Starting model version: ", mv_num)
            saver = tf.train.Saver(max_to_keep=None)
            saver.restore(sess, os.path.join(trainedNet_path, "model.ckpt-" + f"{mv_num}"))

            header = ['model_pred'] + eval_keys
            header.pop(1)  # TODO: ONLY TEMPORARY DELETE BINARY LABEL COULMN
            # header = ['model_pred'] + eval_keys + ['cnn_idx_' + str(i) for i in range(504)]
            csv_path = f"{save_name}_{net_name}_model_{mv_num}.csv"
            csv_handle = open(csv_path, 'w', encoding='UTF8', newline='')
            csv_writer = csv.writer(csv_handle)
            csv_writer.writerow(header)

            while True:
                # running individual batches
                try:
                    if is_msl:
                        pd, cd, e_vars = sess.run([correct_pred, cond_dist, eval_vars])
                        e_vars.pop(0)  # TODO: ONLY TEMPORARY DELETE BINARY LABEL COLUMN
                        n_sounds_perceived = decide_sound_presence(cd, criterion=net_params["decision_criterion"])
                    else:
                        pd, pd_corr, cd, e_vars = sess.run([net_pred, correct_pred, cond_dist, eval_vars])
                    # prepare result to write into .csv
                    csv_rows = list(zip(n_sounds_perceived, *e_vars))
                    # csv_rows = list(zip(pd, *e_vars, cd.tolist()))
                    print("Writing data to file ...")
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
        # search for dense layer weights or posterior
        # all variables
        """
        all_vars = list(set(v.op.name for v in tf.global_variables()).difference([]))

        # retrain variables
        retrain_vars = list()
        for weight in all_vars:
            if weight.find("fc") != -1:
                retrain_vars.append(weight)
            if weight.find("out") != -1:
                retrain_vars.append(weight)
        """
        # get variable list for restoring in saver
        # var_list = tf.contrib.framework.get_variables_to_restore(exclude=None)
        # model_weights = os.path.join(net_name, "model.ckpt-" + model_version[0])
        # ckpt = tf.train.load_checkpoint(model_weights)
        newpath = os.path.join(net_weights + "_MSL/" + net_name)
        display_step = run_params["display_step"]
        sess.run(stim_iter.initializer)
        # saver = tf.train.Saver(max_to_keep=None, var_list=var_list)
        saver = tf.train.Saver(max_to_keep=None)
        learning_curve_acc = []
        learning_curve_auc = []
        errors_count = 0
        step = 1
        mv_num = model_version[0]
        try:
            sess.graph.finalize()
            # sess.run(partially_frozen)
            while True:
                # sess.run([optimizer,check_op])
                try:
                    if step == 1:
                        files = os.listdir(newpath)
                        checkpoint_files = []
                        for file in files:
                            if (file.split("/")[-1]).split(".")[0] == 'model':
                                checkpoint_files.append(os.path.join(newpath, file))
                        # Ensure there is at least one checkpoint file before accessing its name
                        if checkpoint_files.__len__():
                            latest_addition = max(checkpoint_files, key=os.path.getctime)
                            latest_addition_name = latest_addition.split(".")[-2]
                            saver.restore(sess, newpath + "/model." + latest_addition_name)
                            step = int(latest_addition_name.split("-")[1])
                            learning_curve_auc = json.load(open(glob.glob(os.path.join(newpath, "*auc*"))[0]))
                            learning_curve_acc = json.load(open(glob.glob(os.path.join(newpath, "*acc*"))[0]))
                        else:
                            print("No checkpoint files found in the directory.")
                            saver.restore(sess, os.path.join(trainedNet_path, "model.ckpt-" + f"{mv_num}"))
                        # freeze_session(sess, keep_var_names=retrain_vars)  # freeze all layers prior to dense layer
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
                    loss, acc, bl, auc_out, update_auc_out = sess.run([cost, accuracy, data_label['train/binary_label'],
                                                                       auc, update_op_auc])
                    print("Batch Labels: ", bl)
                    print(f"Iter {step * batch_size}, "
                          f"Minibatch Loss = {loss}, "
                          f"Training Accuracy = {acc},"
                          f"AUC = {update_auc_out}")
                    learning_curve_acc.append([int(step * batch_size), float(acc)])
                    learning_curve_auc.append([int(step * batch_size), float(update_auc_out)])
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
                    print("Checkpoint Complete")
                    if is_msl:
                        print("Resetting AUC counter")
                        sess.run(running_vars_auc_initializer)
                # Just for testing the model/call_model
                if step == run_params["total_steps"]:
                    print("Break!")
                    break
                print(f"Current step: {step}")
                step += 1
        except tf.errors.OutOfRangeError:
            print("Out of Range Error. Optimization Finished")
        except tf.errors.DataLossError as e:
            print("Corrupted file found!!")
            pdb.set_trace()
        finally:
            print("Total errors: ", errors_count)
            print("Training stopped.")

        with open(newpath + '/learning_curve_acc.json', 'w') as f:
            json.dump(learning_curve_acc, f)
        with open(newpath + '/learning_curve_auc.json', 'w') as f:
            json.dump(learning_curve_auc, f)

    # cleanup
    sess.close()
    tf.reset_default_graph()



if __name__ == '__main__':
    netweights_path = "/home/max/PycharmProjects/BinauralLocalizationCNN/netweights_MSL/"
    first_net_path = os.path.join(netweights_path, sorted(os.listdir(netweights_path))[0])
    config_fname = 'config_array.npy'
    config_array = np.load(os.path.join(first_net_path, config_fname), allow_pickle=True)
    stim_tfrecs = os.path.join("*test_azi*.tfrecords")
    res_name = os.path.join('Result', 'NumJudge_result')
    run_CNN(stim_tfrecs, first_net_path, res_name)


