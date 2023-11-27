import os
from run_CNN import run_CNN
from MSL.config_MSL import CONFIG_TEST as cfg

# MSL
stim_tfrecs = os.path.join("numjudge_*test_azi*.tfrecords")
# trained net parameters
for net in range(1, 2):
    trainedNet = os.path.join('netweights_MSL', f'net{net}')
    # result same name
    res_name = os.path.join('Result', 'NumJudge_result')
    # run the model
    run_CNN(stim_tfrec_pattern=stim_tfrecs, trainedNet_path=trainedNet, save_name=res_name, cfg=cfg)

