import os
from run_CNN import run_CNN

CONFIG_TEST = dict(DEFAULT_NET_PARAM={'cpu_only': False, 'regularizer': None, "n_classes_localization": 504},
                   DEFAULT_COST_PARAM={"multi_source_localization": False},
                   DEFAULT_RUN_PARAM={'learning_rate': 1e-3,
                                      'batch_size': 16,
                                      'testing': True,
                                      'model_version': [100000]},
                   DEFAULT_DATA_PARAM={"augment": False}
                   )
CONFIG_TEST["DEFAULT_RUN_PARAM"]["testing"] = True
CONFIG_TEST["DEFAULT_RUN_PARAM"]["training"] = False
CONFIG_TEST["DEFAULT_RUN_PARAM"]["validating"] = False


stim_tfrecs_ele = "locaaccu_noise_ele*.tfrecords"
stim_tfrecs_azi = "locaaccu_noise_azi*.tfrecords"

ele = stim_tfrecs_ele.split("_")[-1].split(".")[0].split("*")[0]
azi = stim_tfrecs_azi.split("_")[-1].split(".")[0].split("*")[0]

# trained net parameters
for net in range(5, 11):
    trainedNet = os.path.join('netweights', f'net{net}')
    # result same name
    save_name = os.path.join('Result', f'LocaAccu_noise_{azi}_result')
    # run the model
    run_CNN(stim_tfrec_pattern=stim_tfrecs_azi, trainedNet_path=trainedNet, save_name=save_name, cfg=CONFIG_TEST)

    save_name = os.path.join('Result', f'LocaAccu_noise_{ele}_result')
    # run the model
    run_CNN(stim_tfrec_pattern=stim_tfrecs_ele, trainedNet_path=trainedNet, save_name=save_name, cfg=CONFIG_TEST)


