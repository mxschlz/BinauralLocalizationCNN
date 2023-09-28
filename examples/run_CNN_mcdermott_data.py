import os
from run_CNN import run_CNN

for net in range(1, 11):
    # trained net parameters
    trainedNet = os.path.join('netweights', f'net{net}')
    # stimulus records
    stim_tfrecs = os.path.join('tfrecords', 'mcdermott', 'broadband_noise_elevation*.tfrecords')

    # result same name
    res_name = os.path.join('Result', 'broadband_noise_elevation')
    # run the model
    run_CNN(stim_tfrecs, trainedNet, res_name)

    # ITD/ILD
    stim_itdild = os.path.join('tfrecords', 'mcdermott', 'noise*_ITDILD.tfrecords')
    res_name = os.path.join('Result', 'ITDILD')
    run_CNN(stim_itdild, trainedNet, res_name)

for net in range(4, 11):
    # trained net parameters
    trainedNet = os.path.join('netweights', f'net{net}')
    # stimulus records
    stim_tfrecs = os.path.join('tfrecords', 'mcdermott', 'SL_unmodified_ears.tfrecords')
    # result name
    res_name = os.path.join('Result', 'SL_unmodified_ears')
    # run the model
    run_CNN(stim_tfrecs, trainedNet, res_name)

# bandwidth
# stim_bd = os.path.join('tfrecords', 'generated', 'noise_bandwidth*.tfrecords')
# res_name = os.path.join('Result', 'noise_bw')
# run_CNN(stim_bd, trainedNet, res_name)
