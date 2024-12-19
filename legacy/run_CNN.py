import collections
import csv
import glob
import json
import os
# from CNN_util import freeze_session
import pdb
import time
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gradients

# custom memory saving gradient
import memory_saving_gradients
from CNN_util import build_tfrecords_iterator, get_feature_dict, cost_function
from NetBuilder import NetBuilder
from analysis_and_plotting.decision_rule import decide_sound_presence
from augment import apply_random_augmentation

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
    stim_files = glob.glob(stim_tfrec_pattern)  # find files from pattern
    stim_feature = get_feature_dict(stim_files[0])  # get stimulus features
    stim_dset = build_tfrecords_iterator(stim_tfrec_pattern, stim_feature, **ds_params)  # initialize tf dataset
    is_msl = cfg["DEFAULT_COST_PARAM"]["multi_source_localization"]  # check whether multi-source localization applies
    # from the stim_dset, create a batched data iterator
    batch_size = run_params['batch_size']  # use batch size of 16 (default)
    batch_size_tf = tf.constant(batch_size, dtype=tf.int64)  # make tf constant
    stim_dset = stim_dset.shuffle(buffer_size=batch_size).batch(batch_size=batch_size_tf,
                                                                drop_remainder=True)  # always shuffle
    stim_iter = stim_dset.make_initializable_iterator()
    data_samp = stim_iter.get_next()

    # get a data label dict from the data sample
    data_label = collections.OrderedDict()
    for k, v in data_samp.items():
        if k not in ('train/image', 'train/image_height', 'train/image_width'):
            data_label[k] = data_samp[k]
    augment = ds_params["augment"]  # check whether data augmentation applies
    if augment is True:
        data_samp["train/image"] = apply_random_augmentation(data_samp["train/image"])  # apply random augmentation

    # build the CNN net
    # load config array from trained network
    net_weights, net_name = os.path.split(trainedNet_path)
    config_fname = 'config_array.npy'
    config_array = np.load(os.path.join(trainedNet_path, config_fname), allow_pickle=True)  # weights path

    # network input
    # the signal is power-transformed
    # coch_sig = data_samp['train/image']
    new_sig_nonlin = tf.pow(data_samp['train/image'], 0.3)  # power transform according to Francl 2022

    # build the neural network
    net = NetBuilder(cpu_only=net_params['cpu_only'])  # build the net via CPU only or GPU
    net_out = net.build(config_array, new_sig_nonlin, **net_params)  # build the net

    # regularizer
    if net_params['regularizer'] is not None:  # I don't know actually (didnt use)
        reg_term = tf.contrib.layers. \
            apply_regularization(net_params['regularizer'],
                                 (tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

    # get network cost and labels
    # the multi-source paradigm uses a different cost function because of different data labeling and outputs
    # ignore the first part under if_msl
    if is_msl:  # multi source localization
        cost, net_labels = cost_function(data_samp, net_out, **cost_params)
        cond_dist = tf.nn.sigmoid(net_out)
        auc, update_op_auc = tf.metrics.auc(net_labels, cond_dist, name='auc')
        running_vars_auc = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                             scope='auc')
        running_vars_auc_initializer = tf.variables_initializer(var_list=running_vars_auc)
        # top_k = tf.nn.top_k(net_out, 5)
        # Evaluate model
        correct_pred = tf.equal(tf.to_int64(net_out > 0.5), tf.cast(net_labels, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    elif not is_msl:  # single sound source localization
        cost, net_labels = cost_function(data_samp, net_out, **cost_params)  # get labels and cost function


        # Building tensors on top / after the last tensor of the network:
        cond_dist = tf.nn.softmax(net_out)  # use softmax distribution
        # Evaluate model
        net_pred = tf.argmax(cond_dist, 1)  # get the maximum probability value
        # top_k = tf.nn.top_k(net_out, 5)  # not used
        # correct predictions
        correct_pred = tf.equal(tf.argmax(net_out, 1), tf.cast(net_labels, tf.int64))  # check whether correct or not
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # calculate accuracy

    if net_params['regularizer'] is not None:
        cost = tf.add(cost, reg_term)

    # launch the model
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    if is_msl:  # only retrain fully connected layers
        first_fc_idx = [x.name for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)].index('wc_fc_0:0')
        late_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[first_fc_idx:]
    with tf.control_dependencies(update_ops):
        if is_msl:
            update_grads = (tf.train.AdamOptimizer(learning_rate=run_params['learning_rate'], epsilon=1e-4)
                            .minimize(cost, var_list=late_layers))  # only update late layers
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
    sess.run(init_op)  # run session

    # file handle for .csv writing
    data_label = dict(sorted(data_label.items()))  # sort dict so that the order stays the same over all model results
    eval_keys = list(data_label.keys())  # keys
    eval_vars = list(data_label.values())  # variables

    testing = run_params["testing"]  # whether testing
    validating = run_params["validating"]  # whether validation
    training = run_params["training"]  # whether training
    model_version = run_params['model_version']  # model version 100000 from Francl 2022
    auc_results = dict()
    if validating:  # basically like training but without updating weights
        for mv_num in model_version:
            auc_results[mv_num] = []  # save area under the curve for ROC
            sess.run(stim_iter.initializer)
            # load model
            print("Starting model version: ", mv_num)
            saver = tf.train.Saver(max_to_keep=None)  # instantiate the tf saver
            saver.restore(sess, os.path.join(trainedNet_path, "model.ckpt-" + f"{mv_num}"))  # load weights
            # run augmented test set for model validation
            while True:
                try:
                    # Calculate batch loss and accuracy
                    loss, acc, auc_out, update_auc_out = sess.run([cost, accuracy, auc, update_op_auc])  # run session
                    auc_results[mv_num].append(update_auc_out)  # append the area under the curve value
                    print(f"AUC = {update_auc_out}")
                except tf.errors.OutOfRangeError:
                    print('Dataset finished')
                    break
                finally:
                    pass
        npy_path_auc = f"{save_name}_val_auc.npy"
        np.save(npy_path_auc, auc_results)  # save AUC values
    elif testing:
        for mv_num in model_version:
            sess.run(stim_iter.initializer)  # run the session
            print("Starting model version: ", mv_num)
            saver = tf.train.Saver(max_to_keep=None)  # instantiate saver
            saver.restore(sess, os.path.join(trainedNet_path, "model.ckpt-" + f"{mv_num}"))  # load weights
            header = ['model_pred'] + eval_keys  # make header for the csv file
            # header = ['model_pred'] + eval_keys + ['cnn_idx_' + str(i) for i in range(504)]
            if is_msl:  # delete binary label column (no need to save it)
                bin_label_idx = header.index("train/binary_label")
                header.pop(bin_label_idx)
            csv_path = f"{save_name}_{net_name}_model_{mv_num}.csv"
            csv_handle = open(csv_path, 'w', encoding='UTF8', newline='')  # get csv handle
            csv_writer = csv.writer(csv_handle)
            csv_writer.writerow(header)  # write header
            if is_msl:  # save more in-depth data
                cd_data = list()
                binary_label_data = list()
                cd_path = f"{save_name}_{net_name}_model_{mv_num}_cd.npy"
                binary_label_path = f"{save_name}_{net_name}_model_{mv_num}_binary_labels.npy"
            while True:
                # running individual batches
                try:
                    if is_msl:  # not important
                        pd, cd, e_vars = sess.run([correct_pred, cond_dist, eval_vars])
                        # e_vars = filter_sparse_to_dense(eval_vars)
                        binary_label = e_vars.pop(0)
                        n_sounds_perceived = decide_sound_presence(cd, criterion=net_params["decision_criterion"])
                    else:  # this is important
                        pd, pd_corr, cd, e_vars = sess.run([net_pred, correct_pred, cond_dist, eval_vars])
                        # -> pd is model prediction, e_vars is ground truth

                        # Tensors to evaluate in the session (all have shape (batch_size) so (16), I think):
                        # net_pred is the model prediction: argmax(softmax(net_out)) (where net_out is a logit tensor)
                        # correct_pred is boolean tensor whether the model prediction is correct: equal(argmax(net_out), net_labels)
                        # cond_dist is prob dist of prediction: softmax(net_out)
                        # eval_vars is the ground truth: net_labels -> can't wrap my head around how it's created in the code above...

                        # Output of the session: same shape as input with leaves replaced by the values of the tensors returned by the session

                    if is_msl:  # not important
                        cd_data.append(cd)
                        binary_label_data.append(binary_label)

                        # prepare result to write into .csv
                        csv_rows = list(zip(n_sounds_perceived, *e_vars))
                    else:
                        csv_rows = list(zip(pd, *e_vars))  # make csv rows (16 for each iteration)
                    # csv_rows = list(zip(n_sounds_perceived, *e_vars, cd.tolist()))
                    print("Writing data to file ...")
                    csv_writer.writerows(csv_rows)  # write data
                except tf.errors.ResourceExhaustedError:
                    print("Out of memory error")
                    break
                except tf.errors.OutOfRangeError:
                    print('Dataset finished')
                    break

                finally:
                    pass
            if is_msl:
                np.save(cd_path, np.array(cd_data))
                np.save(binary_label_path, np.array(binary_label_data))

            # close the csv file
            csv_handle.close()
    elif training:
        # search for dense layer weights or posterior
        # get variable list for restoring in saver
        newpath = os.path.join(
            net_weights + "_MSL/" + net_name)  # this folder leads to new directory (change for your paradigm)
        display_step = run_params["display_step"]  # display interim results during training
        sess.run(stim_iter.initializer)
        # saver = tf.train.Saver(max_to_keep=None, var_list=var_list)
        saver = tf.train.Saver(max_to_keep=None)
        learning_curve_acc = []  # accuracy during training
        learning_curve_auc = []  # area under the curve ROC during training (only important for multi-source paradigm)
        errors_count = 0  # count errors
        step = 1  # count training steps
        mv_num = model_version[0]
        try:
            sess.graph.finalize()
            # sess.run(partially_frozen)
            while True:
                # sess.run([optimizer,check_op])
                try:
                    if step == 1:  # the following only searches for previous checkpoints
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
                if step % display_step == 0:  # check whether to display the interim results
                    # Calculate batch loss and accuracy
                    loss, acc, bl, auc_out, update_auc_out = sess.run([cost, accuracy, data_label['train/binary_label'],
                                                                       auc, update_op_auc])
                    print("Batch Labels: ", bl)
                    print(f"Iter {step * batch_size}, "
                          f"Minibatch Loss = {loss}, "
                          f"Training Accuracy = {acc},"
                          f"AUC = {update_auc_out}")
                    learning_curve_acc.append([int(step), float(acc)])  # append acc every display step
                    learning_curve_auc.append([int(step), float(update_auc_out)])  # append AUC ROC every display step
                if step % run_params["checkpoint_step"] == 0:  # save the interim model weights as checkpoints
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
                    if is_msl:  # reset AUC counter
                        print("Resetting AUC counter")
                        sess.run(running_vars_auc_initializer)
                # Just for testing the model/call_model
                if step == run_params["total_steps"]:  # break after total steps
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

        # save additional data
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
