import os

from legacy.run_CNN import run_CNN

CONFIG_TEST = dict(DEFAULT_NET_PARAM={'cpu_only': False, 'regularizer': None, "n_classes_localization": 504},
                   DEFAULT_COST_PARAM={"multi_source_localization": False},
                   DEFAULT_RUN_PARAM={'learning_rate': 1e-3,
                                      'batch_size': 16,
                                      'testing': True,
                                      "training": False,
                                      "validating": False,
                                      'model_version': [100000]},
                   DEFAULT_DATA_PARAM={"augment": False}
                   )

stim_tfrecs = "tfrecords/locaaccu_babble_ele*.tfrecords"

plane = stim_tfrecs.split("_")[-1].split(".")[0].split("*")[0]

# trained net parameters
for net in range(1, 11):
    trainedNet = os.path.join('netweights', f'net{net}')
    # result same name
    save_name = os.path.join('Results_locaaccu', f'LocaAccu_babble_{plane}_result')
    # run the model
    run_CNN(stim_tfrec_pattern=stim_tfrecs, trainedNet_path=trainedNet, save_name=save_name, cfg=CONFIG_TEST)
