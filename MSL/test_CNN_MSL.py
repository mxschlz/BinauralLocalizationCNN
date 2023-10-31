import os
from run_CNN import run_CNN
from MSL.config_MSL import CONFIG_TEST as cfg


# LocaAccu tfrecs
stim_tfrecs = os.path.join('tfrecords', 'msl', "locaaccu_noise_v.tfrecords")
# trained net parameters
for net in range(1, 11):
    trainedNet = os.path.join('netweights', f'net{net}')
    # result same name
    res_name = os.path.join('Result', 'locaaccu_noise_v')
    # run the model
    run_CNN(stim_tfrecs, trainedNet, res_name)

# MSL
stim_tfrecs = os.path.join("numjudge_*test_azi*.tfrecords")
# trained net parameters
for net in range(1, 11):
    trainedNet = os.path.join('netweights_MSL', f'net{net}')
    # result same name
    res_name = os.path.join('Result', 'NumJudge_result')
    # run the model
    run_CNN(stim_tfrec_pattern=stim_tfrecs, trainedNet_path=trainedNet, save_name=res_name, cfg=cfg)

