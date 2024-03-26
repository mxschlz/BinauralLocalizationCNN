import os
from run_CNN import run_CNN

CONFIG_TEST = dict(DEFAULT_NET_PARAM={'cpu_only': False, 'regularizer': None, "n_classes_localization": 504},
                   DEFAULT_COST_PARAM={"multi_source_localization": False},
                   DEFAULT_RUN_PARAM={'learning_rate': 1e-3,
                                      'batch_size': 16,
                                      'testing': True,
                                      "validating": False,
                                      "training": False,
                                      'model_version': [100000]},
                   DEFAULT_DATA_PARAM={"augment": False}
                   )

for net in range(1, 11):
    # trained net parameters
    trainedNet = os.path.join('netweights', f'net{net}')
    # stimulus records
    stim_tfrecs = os.path.join('tfrecords', 'mcdermott', 'broadband_noise_azimuth*.tfrecords')

    # result same name
    res_name = os.path.join('Result', 'broadband_noise_azimuth')
    # run the model
    run_CNN(stim_tfrecs, trainedNet, cfg=CONFIG_TEST, save_name=res_name)
