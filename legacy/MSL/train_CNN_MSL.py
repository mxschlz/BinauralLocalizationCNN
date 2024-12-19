import os

import legacy.MSL.config_MSL as cfg
from legacy.run_CNN import run_CNN

stim_tfrecs = os.path.join("tfrecords/*train*.tfrecords")
total_retries = (cfg.CONFIG_TRAIN["DEFAULT_RUN_PARAM"]["total_steps"] /
                 cfg.CONFIG_TRAIN["DEFAULT_RUN_PARAM"]["checkpoint_step"])
cfg.CONFIG_TRAIN["DEFAULT_RUN_PARAM"]["testing"] = False
cfg.CONFIG_TRAIN["DEFAULT_RUN_PARAM"]["training"] = True
cfg.CONFIG_TRAIN["DEFAULT_RUN_PARAM"]["validating"] = False

# trained net parameters
for net in range(1, 11):
    retry_count = 0
    while True:  # continue training until total retries is reached
        print(f"### START NETWORK {net} ###")
        trainedNet = os.path.join('netweights', f'net{net}')
        # run the model
        run_CNN(stim_tfrec_pattern=stim_tfrecs, trainedNet_path=trainedNet, cfg=cfg.CONFIG_TRAIN)
        retry_count += 1
        if retry_count >= total_retries:
            break
