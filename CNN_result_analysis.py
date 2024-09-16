import csv
import glob
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import slab
from scipy import stats as sstats


def read_resfiles(file_pattern, filetype="csv", extra_lbs=None):
    """
    read result files generated from the CNN model
    :param file_pattern: regrexp pattern to look for result files
    :param extra_lbs: labels to add as a new column for each individual data file. must have the same
                      length as the files found by the file_pattern, and must be numerical
    :return: header and data as lists
    """

    header, data = [], []
    files = glob.glob(file_pattern)

    if extra_lbs is not None:
        assert len(files) == len(extra_lbs), "number of labels must match number of files"

    if filetype == "csv":
        # read header
        with open(files[0], newline='') as csvfile:
            csv_rd = csv.reader(csvfile)
            header = next(csv_rd)

        # read data
        for idx, f in enumerate(files):
            fdata = np.genfromtxt(f, delimiter=',')[1:]
            if extra_lbs is not None:
                # add extra label into the data
                lb_array = extra_lbs[idx] * np.ones((fdata.shape[0], 1))
                fdata = np.concatenate([fdata, lb_array], axis=1)
            data.append(fdata)

        return header, data

    elif filetype == "npy":
        for idx, f in enumerate(files):
            fdata = np.load(f, allow_pickle=True)
            data.append(fdata)
        return data


def CNNpos_to_loc(CNN_pos, bin_size=5):
    """
    convert bin label in the CNN from Francl 2022 into [azim, elev] positions
    :param CNN_pos: int, [0, 503]
    :param bin_size: int, degree. note that elevation bin size is 2*bin_size
    :return: tuple, (azi, ele)
    """
    n_azim = int(360 / bin_size)
    bin_idx = divmod(CNN_pos, n_azim)

    return bin_size * bin_idx[1], bin_size * 2 * bin_idx[0]


def result_figure1(model_data_patt, human_data_path):
    """
    analyze and plot figure 1
    :param model_data_patt: regrexp pattern to read result files
    :param human_data_path: data file from Francl 2022 paper
    :return:
    """
    # human data
    human_data_df = pd.read_excel(human_data_path, sheet_name='1f')
    human_data_sim = _simulate_f1_humandata(human_data_df)

    # model data
    md_header, md_data = read_resfiles(model_data_patt)
    md_data = np.concatenate(md_data, axis=0)
    # plot actual azim loc vs. predicted azim loc
    col_act = md_header.index('train/cnn_idx')
    azim_act = CNNpos_to_loc(md_data[:, col_act])[1]
    azim_act[azim_act > 180] = azim_act[azim_act > 180] - 360
    col_pred = md_header.index('model_pred')
    azim_pred = CNNpos_to_loc(md_data[:, col_pred])[1]
    azim_pred[azim_pred > 180] = azim_pred[azim_pred > 180] - 360
    # collapse front and back
    azim_pred[azim_pred > 90] = 180 - azim_pred[azim_pred > 90]
    azim_pred[azim_pred < -90] = -180 - azim_pred[azim_pred < -90]
    # prepare model data for violin plot
    md_x = np.unique(azim_act)
    md_y = []
    for x in md_x:
        # convert to probability density
        md_y.append(azim_pred[azim_act == x])

    # plotting
    fig, ax = plt.subplots(1, 2)
    ax[0].violinplot(human_data_sim[2], human_data_sim[0], widths=10)
    ax[0].set_title("human performance")
    ax[0].set_xlabel("actual azim position (degrees)")
    ax[0].set_ylabel("judged azim position (degrees)")
    ax[1].violinplot(md_y, md_x, widths=10)
    ax[1].set_title("model performance")
    ax[1].set_xlabel("actual azim position (degrees)")


def _simulate_f1_humandata(data_excel, ns_bin=100):
    """
    Simulates data for Figure 1 based on the excel data sheet in Francl 2022 paper.

    Args:
        data_excel (DataFrame): tfrecords from the excel.
        ns_bin (int): Number of data points to simulate in each position.

    Returns:
        tuple: A tuple containing:
            - x_pos (ndarray): Sorted unique values of the 'Actual Position (Degrees)' column.
            - y_bins (ndarray): Bins representing positions.
            - data_sim (list): Simulated data for each position.
    """

    # Extract unique bins from the 'Predicted Position (Degrees)' column and sort them
    bins = np.sort(np.unique(data_excel['Predicted Position (Degrees)'])).astype(np.float64)

    # Calculate the size of each bin
    bin_size = bins[1] - bins[0]

    # Create bins for the y-axis
    y_bins = np.zeros(bins.shape[0] + 1)
    y_bins[:-1] = bins - bin_size / 2
    y_bins[-1] = bins[-1] + bin_size / 2

    # Extract unique values of 'Actual Position (Degrees)' column and sort them
    x_pos = np.sort(np.unique(data_excel['Actual Position (Degrees)']))

    # Simulate data for each position
    data_sim = []
    for x in x_pos:
        data_x = []
        # Subset the data for the current position and sort it by 'Predicted Position (Degrees)'
        df_sub = data_excel[data_excel['Actual Position (Degrees)'] == x].sort_values('Predicted Position (Degrees)')

        # Generate data from a uniform distribution based on the 'Percentage Responses' column
        for idx, nd in enumerate(df_sub['Percentage Responses']):
            n_tosim = int(ns_bin / 100 * nd)
            data_x.extend((bin_size * np.random.rand(n_tosim) + y_bins[idx]).tolist())

        data_sim.append(data_x)

    return x_pos, y_bins, data_sim


def result_figure2(model_data_patt):
    label, md_data = read_resfiles(model_data_patt)
    md_data = np.concatenate(md_data, axis=0)

    # actual and predicted azim pos
    col_act = label.index('train/cnn_idx')
    azim_act = CNNpos_to_loc(md_data[:, col_act])[1]
    col_pred = label.index('model_pred')
    azim_pred = CNNpos_to_loc(md_data[:, col_pred])[1]

    # get ITD/ILD bias, and low vs high fq conditions
    cond = md_data[:, label.index('train/center_freq')]
    itd_bias = md_data[:, label.index('train/ITD')]
    ild_bias = md_data[:, label.index('train/ILD')]
    itd_vals = np.sort(np.unique(itd_bias))
    ild_vals = np.sort(np.unique(ild_bias))
    # use slab to calculate perceived bias
    cfs = np.sort(np.unique(cond))
    bw = np.unique(md_data[:, label.index('train/bandwidth')])
    sr = md_data[0, label.index('train/sampling_rate')]
    # example signal used to calculate perceived bias
    sig = slab.Sound.whitenoise(duration=1.0, samplerate=sr)
    sig = slab.Binaural(sig)
    exp_sig = []
    for cf in cfs:
        fl = [cf / 2 ** (bw[0] / 2), cf * 2 ** (bw[0] / 2)]
        filt = slab.Filter.band('bp', frequency=tuple(fl),
                                samplerate=sr, length=2048)
        exp_sig.append(filt.apply(sig))
    # calculation
    ils = slab.Binaural.make_interaural_level_spectrum()
    res_itd, res_ild = [], []
    for cf in cfs:
        itd_cf = []
        for itd in itd_vals:
            idx = np.logical_and(itd_bias == itd, cond == cf)
            idx = np.logical_and(idx, ild_bias == 0)
            judged_pos = azim_pred[idx]
            judged_pos[judged_pos > 180] = judged_pos[judged_pos > 180] - 360
            actual_pos = azim_act[idx]
            actual_pos[actual_pos > 180] = actual_pos[actual_pos > 180] - 360
            # observed itd bias
            itd_cf.append([1e6 * (slab.Binaural.azimuth_to_itd(pj, cf) -
                                  slab.Binaural.azimuth_to_itd(pa, cf))
                           for pj, pa in zip(judged_pos, actual_pos)])

        ild_cf = []
        for ild in ild_vals:
            idx = np.logical_and(ild_bias == ild, cond == cf)
            idx = np.logical_and(idx, itd_bias == 0)
            judged_pos = azim_pred[idx]
            judged_pos[judged_pos > 180] = judged_pos[judged_pos > 180] - 360
            actual_pos = azim_act[idx]
            actual_pos[actual_pos > 180] = actual_pos[actual_pos > 180] - 360
            ild_cf.append([slab.Binaural.azimuth_to_ild(pj, cf, ils) -
                           slab.Binaural.azimuth_to_ild(pa, cf, ils)
                           for pj, pa in zip(judged_pos, actual_pos)])

        res_ild.append(deepcopy(ild_cf))
        res_itd.append(deepcopy(itd_cf))

    # plotting
    fig, ax = plt.subplots(2, 2)
    ax_ct = 0
    for cf in cfs[::-1]:
        cf_idx = list(cfs).index(cf)
        for x, ys in zip([itd_vals, ild_vals], [res_itd[cf_idx], res_ild[cf_idx]]):
            ax_r, ax_c = divmod(ax_ct, 2)
            y_mean = [np.array(y).mean() for y in ys]
            y_se = [np.array(y).std() / len(y) ** 0.5 for y in ys]
            # plot
            ax[ax_r][ax_c].errorbar(x[::-1], y_mean, y_se)
            if ax_ct in (0, 2):
                ax[ax_r][ax_c].set_ylim([-600, 600])
                ax[ax_r][ax_c].set_xlim([-750, 750])
                ax[ax_r][ax_c].set_ylabel(r"Response ITD bias ($\mu$s)")
                if ax_ct == 0:
                    ax[ax_r][ax_c].set_title('ITD bias')
                else:
                    ax[ax_r][ax_c].set_xlabel(r"imposed ITD bias ($\mu$s)")
            if ax_ct in (1, 3):
                ax[ax_r][ax_c].set_ylim([-40, 40])
                ax[ax_r][ax_c].set_ylabel("Response ILD bias (dB)")
                if ax_ct == 1:
                    ax[ax_r][ax_c].set_title('ILD bias')
                else:
                    ax[ax_r][ax_c].set_xlabel("imposed ILD bias (dB)")

            ax_ct += 1


def result_figure3(model_data_patt, human_data_path):
    label, md_data = read_resfiles(model_data_patt)
    md_data = np.concatenate(md_data, axis=0)

    # actual and predicted azim pos
    col_act = label.index('train/cnn_idx')
    azim_act = CNNpos_to_loc(md_data[:, col_act])[1]
    col_pred = label.index('model_pred')
    azim_pred = CNNpos_to_loc(md_data[:, col_pred])[1]
    azim_act[azim_act > 180] = azim_act[azim_act > 180] - 360
    azim_pred[azim_pred > 180] = azim_pred[azim_pred > 180] - 360
    # collapse front and back
    azim_pred[azim_pred > 90] = 180 - azim_pred[azim_pred > 90]
    azim_pred[azim_pred < -90] = -180 - azim_pred[azim_pred < -90]
    # mean abs bias as function of azimuth
    azim_vals = np.sort(np.unique(azim_act))
    azim_abs_bias = []
    azim_abs_bias_sem = []
    for av in azim_vals:
        idx = azim_act == av
        err_abs = np.abs(azim_pred[idx] - azim_act[idx])
        azim_abs_bias.append(err_abs.mean())
        azim_abs_bias_sem.append(sstats.bootstrap([err_abs], np.mean, n_resamples=10).standard_error)

    # rms error as a function of bandwidth
    col_bd = label.index('train/bandwidth')
    bd_data = md_data[:, col_bd]
    bd_vals = np.sort(np.unique(bd_data))
    rms_bd = []
    rms_bd_sem = []
    for bd in bd_vals:
        idx = bd_data == bd
        rms = np.sqrt((azim_pred[idx] - azim_act[idx]) ** 2)
        rms_bd.append(rms.mean())
        rms_bd_sem.append(sstats.bootstrap([rms], np.mean, n_resamples=10).standard_error)

    # load human data
    # mean abs bias as function of azimuth
    hd_df1 = pd.read_excel(human_data_path, sheet_name='3b')
    hd_1x = hd_df1["Broadband Noise"][1:].to_numpy(dtype=np.float64)
    hd_1ym = hd_df1["Broadband Noise.1"][1:].to_numpy(dtype=np.float64)
    hd_1ye = hd_df1["Broadband Noise Top Error.1"][1:].to_numpy(dtype=np.float64)
    hd_1ye = hd_1ye - hd_1ym
    # rms error as a function of bandwidth
    hd_df2 = pd.read_excel(human_data_path, sheet_name='3e')
    hd_2 = hd_df2[hd_df2['frequency'] == 2000].sort_values("Bandwidth (Octaves)")

    # plotting
    fig, ax = plt.subplots(2, 2)
    # human, dprime vs. azimuth
    ax[0][0].errorbar(hd_1x, hd_1ym, hd_1ye)
    ax[0][0].invert_yaxis()
    ax[0][0].set_xlabel('source azimuth (degree)')
    ax[0][0].set_ylabel("d' (flipped)")
    ax[0][0].set_title('Human')
    # model
    ax[0][1].errorbar(azim_vals, azim_abs_bias, azim_abs_bias_sem)
    ax[0][1].set_xlabel('source azimuth (degree)')
    ax[0][1].set_ylabel('mean abs. error (degree)')
    ax[0][1].set_title('Model')
    # human, rms vs bandwidth
    ax[1][0].errorbar(hd_2["Bandwidth (Octaves)"],
                      hd_2["RMS Error (Degrees)"],
                      hd_2["standard deviation"])
    ax[1][0].set_xlabel('Bandwidth (octaves)')
    ax[1][0].set_ylabel('r.m.s. error (degree)')
    # model
    ax[1][1].errorbar(bd_vals, rms_bd, rms_bd_sem)
    ax[1][1].set_xlabel('Bandwidth (octaves)')
    ax[1][1].set_ylabel('r.m.s. error (degree)')


def result_figure4(model_data_patt):
    header, data = read_resfiles(model_data_patt)  # read results file
    data = np.concatenate(data, axis=0)  # concatenate data
    col_act = header.index('train/cnn_idx')  # get actual position column
    col_pred = header.index('model_pred')  # get predicted position
    loc_act = CNNpos_to_loc(data[:, col_act])[1]  # convert bin to azi, ele positions
    loc_pred = CNNpos_to_loc(data[:, col_pred])[1]


if __name__ == '__main__':
    import scipy

    results_root = "Result"
    model_data_patt = os.path.join(results_root, "*_azi*model_333.csv")
    header, data = read_resfiles(model_data_patt)  # read results file
    data = np.concatenate(data, axis=0)  # concatenate data

    # get azi, ele coordinates
    col_act = header.index('train/cnn_idx')  # get actual position column
    col_pred = header.index('model_pred')  # get predicted position

    col_act = header.index('train/cnn_idx')
    col_pred = header.index('model_pred')
    azim_act = CNNpos_to_loc(data[:, col_act])[0]
    azim_pred = CNNpos_to_loc(data[:, col_pred])[0]

    # collapse front and back
    azim_pred[azim_pred > 90] = 180 - azim_pred[azim_pred > 90]
    azim_pred[azim_pred < -90] = -180 - azim_pred[azim_pred < -90]

    ele_act = CNNpos_to_loc(data[:, col_act])[1]
    ele_pred = CNNpos_to_loc(data[:, col_pred])[1]

    targets = np.array([[azi, ele] for azi, ele in zip(azim_act, ele_act)])
    responses = np.array([[azi, ele] for azi, ele in zip(azim_pred, ele_pred)])

    azimuths = np.unique(targets[:, 0])
    elevations = np.unique(targets[:, 1])

    elevation_gain, n = scipy.stats.linregress(targets[:, 1], responses[:, 1])[:2]
    rmse = np.sqrt(np.mean(np.square(targets - responses), axis=0))
    variability = np.mean([np.std(responses[np.where(np.all(targets == target, axis=1))], axis=0) for target in
                           np.unique(targets, axis=0)], axis=0)
    az_rmse, ele_rmse = rmse[0], rmse[1]
    az_var, ele_var = variability[0], variability[1]
