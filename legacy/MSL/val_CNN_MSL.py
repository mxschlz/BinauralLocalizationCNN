import os

from legacy.MSL.config_MSL import CONFIG_TEST as cfg
from legacy.run_CNN import run_CNN

# MSL
stim_tfrecs = os.path.join("tfrecords/numjudge_*test_azi*.tfrecords")
plane = stim_tfrecs.split("_")[-1].split("*")[0]
cfg["DEFAULT_RUN_PARAM"]["testing"] = False
cfg["DEFAULT_RUN_PARAM"]["training"] = False
cfg["DEFAULT_RUN_PARAM"]["validating"] = True

cfg["DEFAULT_DATA_PARAM"]["augment"] = False

# trained net parameters
for net in range(5, 11):
    trainedNet = os.path.join('netweights_MSL', f'net{net}')
    # run the model
    run_CNN(stim_tfrec_pattern=stim_tfrecs,
            trainedNet_path=trainedNet,
            cfg=cfg,
            save_name=f"net_{net}")
