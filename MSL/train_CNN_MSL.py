import os
from run_CNN import run_CNN
import MSL.config_MSL as cfg

stim_tfrecs = os.path.join("*train*.tfrecords")
# trained net parameters
for net in range(1, 2):
    trainedNet = os.path.join('netweights', f'net{net}')
    # run the model
    run_CNN(stim_tfrec_pattern=stim_tfrecs, trainedNet_path=trainedNet, cfg=cfg.CONFIG_TRAIN)
