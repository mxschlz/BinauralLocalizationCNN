import warnings

import tensorflow as tf


class NetBuilder:

    def __init__(self, cpu_only=False):
        self.input = None
        self.input1 = 0
        self.input2 = 0
        self.layer = 0
        self.layer1 = 0
        self.layer2 = 0
        self.layer_fc = 0
        self.layer_fc1 = 0
        self.layer_fc2 = 0
        self.layer_out = 0
        self.layer_out1 = 0
        self.layer_out2 = 0
        self.fc_out = 0
        self.cpu_only = cpu_only

    def build(self, config_array, subbands_batch, training_state=True,
              dropout_training_state=True, filter_dtype=tf.float32, padding='VALID',
              n_classes_localization=504, branched=False, regularizer=None, **kwargs):

        if branched:
            warnings.warn("'branched' is not implemented. setting it to False")
            branched = False

        # net_input=tf.constant(1., shape=[16,72,30000,1],dtype=filter_dtype)
        # self.input=net_input
        self.input = subbands_batch
        self.input1 = 0
        self.input2 = 0
        self.layer = 0
        self.layer1 = 0
        self.layer2 = 0
        self.layer_fc = 0
        self.layer_fc1 = 0
        self.layer_fc2 = 0
        self.layer_out = 0
        self.layer_out1 = 0
        self.layer_out2 = 0
        start_point = 0
        branched_point = False
        second_branch = [["/gpu:2"], ["/gpu:3"]]
        gpu2 = second_branch[0][0]
        for lst in config_array:
            # TODO: change to use cpu only if no GPU on the system
            if not self.cpu_only:
                gpu1 = lst[0][0]
            else:
                gpu1 = '/CPU:0'
            start_point = 1
            if branched_point is False:
                with tf.device(gpu1):
                    for element in lst[1:]:
                        if element[0] == 'conv':
                            size = self.input.get_shape()
                            kernel_size = [element[1][0], element[1][1], size[3], element[1][2]]
                            stride_size = [1, element[2][0], element[2][1], 1]
                            filter_height = kernel_size[0]
                            in_height = int(size[1])
                            weight = tf.get_variable("wc_{}".format(self.layer), kernel_size, filter_dtype,
                                                     regularizer=regularizer)
                            bias = tf.get_variable("wb_{}".format(self.layer), element[1][2], filter_dtype)
                            if in_height % stride_size[1] == 0:
                                pad_along_height = max(filter_height - stride_size[1], 0)
                            else:
                                pad_along_height = max(filter_height - (in_height % stride_size[1]), 0)
                            pad_top = pad_along_height // 2
                            pad_bottom = pad_along_height - pad_top
                            if pad_along_height == 0 or padding == 'SAME':
                                weight = tf.nn.conv2d(self.input, weight,
                                                      strides=stride_size, padding=padding)
                            else:
                                paddings = tf.constant([[0, 0], [pad_top, pad_bottom], [0, 0], [0, 0]])
                                input_padded = tf.pad(self.input, paddings)
                                weight = tf.nn.conv2d(input_padded, weight,
                                                      strides=stride_size, padding=padding)

                            self.input = tf.nn.bias_add(weight, bias)
                            self.layer += 1
                            print(element)
                            print(self.input)

                        elif element[0] == 'bn':
                            self.input = tf.layers.batch_normalization(self.input, training=training_state)
                            print(element)
                            print(self.input)

                        elif element[0] == 'relu':
                            self.input = tf.nn.relu(self.input)
                            print(element)
                            print(self.input)

                        elif element[0] == 'pool':
                            self.input = tf.nn.max_pool(self.input, ksize=[1, element[1][0], element[1][1], 1],
                                                        strides=[1, element[1][0], element[1][1], 1],
                                                        padding=padding)
                            tf.add_to_collection('checkpoints', self.input)
                            print(element)
                            print(self.input)

                        elif element[0] == 'fc':
                            dim = self.input.get_shape()
                            wd1 = tf.get_variable("wc_fc_{}".format(self.layer_fc),
                                                  [dim[3] * dim[1] * dim[2], element[1]],
                                                  filter_dtype, regularizer=regularizer)
                            dense_bias1 = tf.get_variable("wb_fc_{}".format(self.layer_fc1), element[1],
                                                          filter_dtype)
                            pool_flat = tf.reshape(self.input, [-1, wd1.get_shape().as_list()[0]])
                            fc1 = tf.add(tf.matmul(pool_flat, wd1), dense_bias1)
                            self.input = tf.cast(fc1, tf.float32)

                            self.layer_fc += 1
                            print(element)
                            print(self.input)

                        elif element[0] == 'fc_bn':
                            self.input = tf.cast(tf.layers.batch_normalization(self.input, training=training_state),
                                                 filter_dtype)
                            print(element)
                            print(self.input)

                        elif element[0] == 'fc_relu':
                            self.input = tf.nn.relu(self.input)
                            print(element)
                            print(self.input)

                        elif element[0] == 'dropout':
                            self.input = tf.layers.dropout(self.input, training=dropout_training_state)
                            print(element)
                            print(self.input)

                        elif element[0] == 'out':
                            dim_1 = self.input.get_shape()
                            w_out = tf.get_variable("wc_out_{}".format(self.layer_out),
                                                    [dim_1[1], n_classes_localization],
                                                    filter_dtype, regularizer=regularizer)
                            b_out = tf.get_variable("wb_out_{}".format(self.layer_out),
                                                    [n_classes_localization], filter_dtype)
                            out = tf.add(tf.matmul(self.input, w_out), b_out)
                            self.input = tf.cast(out, tf.float32)

                            self.layer_out += 1
                            print(element)
                            print(self.input)
        else:
            return self.input


if __name__ == '__main__':
    import os
    import numpy as np
    from CNN_util import get_feature_dict, build_tfrecords_iterator
    import glob

    netweights_path = "/home/max/Projects/BinauralLocalizationCNN/netweights/"
    first_net_path = os.path.join(netweights_path, sorted(os.listdir(netweights_path))[0])
    config_fname = 'config_array.npy'
    config_array = np.load(os.path.join(first_net_path, config_fname), allow_pickle=True)
    stim_tfrec_pattern = os.path.join(os.getcwd(), "tfrecords/msl/numjudge_full_set_talkers_clear_train.tfrecords")
    stim_files = glob.glob(stim_tfrec_pattern)
    stim_feature = get_feature_dict(stim_files[0])
    ds_params = {}
    stim_dset = build_tfrecords_iterator(stim_tfrec_pattern, stim_feature, **ds_params)
    batch_size = tf.constant(16, dtype=tf.int64)
    stim_dset = stim_dset.shuffle(buffer_size=16). \
        batch(batch_size=batch_size, drop_remainder=True)
    stim_iter = stim_dset.make_initializable_iterator()
    data_samp = stim_iter.get_next()
    new_sig_nonlin = tf.pow(data_samp['train/image'], 0.3)
    net_params = {"cpu_only": True}
    net = NetBuilder(**net_params)
    out = net.build(config_array=config_array, subbands_batch=new_sig_nonlin)
