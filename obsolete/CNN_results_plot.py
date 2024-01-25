import matplotlib
matplotlib.use("TkAgg")
from CNN_result_analysis import read_resfiles, CNNpos_to_loc
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from analysis_and_plotting.misc import *


def results_locaaccu(model_data_pattern, human_data_path, plane, stimtype, plot_type="raincloud"):
    if plane == "v":
        dfv = pd.read_hdf(human_data_path)
        filledv = dfv["mode"].ffill()
        data = dfv[np.where(filledv == stimtype, True, False)]  # True where reversed_speech is True
        ele_act = replace_in_array(get_elevation_from_df(data.actual))
        ele_pred = replace_in_array(get_elevation_from_df(data.perceived))
        # load model data
        header, data = read_resfiles(model_data_pattern)  # read results file
        data = np.concatenate(data, axis=0)  # concatenate data
        col_act = header.index('train/cnn_idx')  # get actual position column
        col_pred = header.index('model_pred')  # get predicted position
        loc_act = CNNpos_to_loc(data[:, col_act])[1]  # convert bin to azi, ele positions
        loc_pred = CNNpos_to_loc(data[:, col_pred])[1]
        fig, ax = plt.subplots(1, 2, sharey=True)
        if plot_type == "violin":
            sns.violinplot(x=ele_act, y=ele_pred, color="skyblue", ax=ax[0])
            sns.violinplot(x=loc_act, y=loc_pred, color="skyblue", ax=ax[1])
        elif plot_type == "reg":
            sns.regplot(x=ele_act, y=ele_pred, color="skyblue", ax=ax[0])
            sns.regplot(x=loc_act, y=loc_pred, color="skyblue", ax=ax[1])
        elif plot_type == "raincloud":
            pt.RainCloud(x=ele_act, y=ele_pred, data=dfv, pointplot=True, move=0.2, linecolor="black", ax=ax[0])
            pt.RainCloud(x=loc_act, y=loc_pred, pointplot=True, move=0.2, linecolor="black", ax=ax[1])
        ax[0].set_title("Human performance")
        ax[0].set_xlabel("Actual Elevation")
        ax[0].set_ylabel("Perceived Elevation")
        ax[1].set_title("Model performance")
        ax[1].set_xlabel("Actual Elevation")
        ax[1].set_ylabel("Perceived Elevation")
    if plane == "h":
        dfh = pd.read_hdf(human_data_path)
        filledh = dfh["mode"].ffill()
        data = dfh[np.where(filledh == stimtype, True, False)]  # True where reversed_speech is True
        azi_act = replace_in_array(get_azimuth_from_df(data.actual))
        azi_pred = replace_in_array(get_azimuth_from_df(data.perceived))
        # load model data
        header, data = read_resfiles(model_data_pattern)  # read results file
        data = np.concatenate(data, axis=0)  # concatenate data
        col_act = header.index('train/cnn_idx')  # get actual position column
        col_pred = header.index('model_pred')  # get predicted position
        loc_act = CNNpos_to_loc(data[:, col_act])[0]  # convert bin to azi, ele positions
        loc_pred = CNNpos_to_loc(data[:, col_pred])[0]
        loc_act[loc_act > 180] = loc_act[loc_act > 180] - 360
        loc_pred[loc_pred > 180] = loc_pred[loc_pred > 180] - 360
        # collapse front and back
        loc_pred[loc_pred > 90] = 180 - loc_pred[loc_pred > 90]
        loc_pred[loc_pred < -90] = -180 - loc_pred[loc_pred < -90]
        fig, ax = plt.subplots(1, 2, sharey=True)
        if plot_type == "violin":
            sns.violinplot(x=azi_act, y=azi_pred, color="skyblue", ax=ax[0])
            sns.violinplot(x=loc_act, y=loc_pred, color="skyblue", ax=ax[1])
        elif plot_type == "reg":
            sns.regplot(x=azi_act, y=azi_pred, color="skyblue", ax=ax[0])
            sns.regplot(x=loc_act, y=loc_pred, color="skyblue", ax=ax[1])
        elif plot_type == "raincloud":
            pt.RainCloud(x=azi_act, y=azi_pred, data=dfh, pointplot=True, move=0.2, linecolor="black", ax=ax[0])
            pt.RainCloud(x=loc_act, y=loc_pred, pointplot=True, move=0.2, linecolor="black", ax=ax[1])
        ax[0].set_title("Human performance")
        ax[0].set_xlabel("Actual Azimuth")
        ax[0].set_ylabel("Perceived Azimuth")
        ax[1].set_title("Model performance")
        ax[1].set_xlabel("Actual Azimuth")
        ax[1].set_ylabel("Perceived Azimuth")

