import os
from run_CNN import run_CNN
import MSL.config_MSL as cfg


stim_tfrecs = os.path.join("*train_azi*.tfrecords")
total_retries = (cfg.CONFIG_TRAIN["DEFAULT_RUN_PARAM"]["total_steps"] /
                 cfg.CONFIG_TRAIN["DEFAULT_RUN_PARAM"]["checkpoint_step"])
cfg["DEFAULT_RUN_PARAM"]["testing"] = False
cfg["DEFAULT_RUN_PARAM"]["training"] = True
cfg["DEFAULT_RUN_PARAM"]["validating"] = False

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
