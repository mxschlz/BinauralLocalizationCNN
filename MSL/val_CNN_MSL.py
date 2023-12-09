import os
from run_CNN import run_CNN
from MSL.config_MSL import CONFIG_TEST as cfg

# MSL
stim_tfrecs = os.path.join("numjudge_*test_azi*.tfrecords")
plane = stim_tfrecs.split("_")[-1].split("*")[0]
run_val = True


# trained net parameters
for net in range(1, 11):
    trainedNet = os.path.join('netweights_MSL', f'net{net}')
    # run the model
    run_CNN(stim_tfrec_pattern=stim_tfrecs, trainedNet_path=trainedNet, cfg=cfg, ds_params={"augment": True},
            run_params={"run_val": run_val})
