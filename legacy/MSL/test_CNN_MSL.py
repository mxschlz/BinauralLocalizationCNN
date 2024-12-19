import os

from legacy.MSL.config_MSL import CONFIG_TEST as cfg
from legacy.run_CNN import run_CNN

# MSL
stim_tfrecs = os.path.join("tfrecords/numjudge_*reversed_test_ele*.tfrecords")
plane = stim_tfrecs.split("_")[-1].split("*")[0]
cfg["DEFAULT_RUN_PARAM"]["testing"] = True
cfg["DEFAULT_RUN_PARAM"]["training"] = False
cfg["DEFAULT_RUN_PARAM"]["validating"] = False

# trained net parameters
for net in range(1, 11):
    trainedNet = os.path.join('netweights_MSL', f'net{net}')
    # result same name
    save_name = os.path.join('Results_no_augment_reversed', f'NumJudge_{plane}_result')
    # run the model
    run_CNN(stim_tfrec_pattern=stim_tfrecs, trainedNet_path=trainedNet, save_name=save_name, cfg=cfg)
