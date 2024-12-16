import csv
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import scipy
from matplotlib.collections import LineCollection


def localization_accuracy(data, show=True, plot_dim=2, binned=True, axis=None, show_single_responses=True,
                          elevation='all', azimuth='all'):
    # retrieve data
    azimuths = np.unique(data[:, 0])
    elevations = np.unique(data[:, 1])
    # azimuths = np.array(range(-180, 176, 5))#np.unique(data[:, 0])
    # elevations = np.array(range(0, 61, 10))#np.unique(data[:, 1])
    targets = data[:, :2]  # [az, ele], first two columns
    responses = data[:, 2:]  # [az, ele], last two columns

    # Reverse engineering the shape of data
    # It must be a 2D array with shape (n, 4) where n is the number of trials

    #  elevation gain, rmse, response variability
    elevation_gain, n = scipy.stats.linregress(targets[:, 1], responses[:, 1])[:2]
    rmse = np.sqrt(np.mean(np.square(targets - responses), axis=0))
    variability = np.mean([np.std(responses[np.where(np.all(targets == target, axis=1))], axis=0)
                           for target in np.unique(targets, axis=0)], axis=0)
    az_rmse, ele_rmse = rmse[0], rmse[1]
    az_sd, ele_sd = variability[0], variability[1]
    az_var, ele_var = az_sd ** 2, ele_sd ** 2

    # mean perceived location for each target speaker
    i = 0
    mean_loc = np.zeros(
        (len(azimuths) * len(elevations), 2, 2))  # For each location x, y for speaker and x, y for prediction
    for azimuth in azimuths:
        for elevation in elevations:
            [perceived_targets] = data[np.where(np.logical_and(data[:, 0] == azimuth, data[:, 1] == elevation)), 2:]
            if perceived_targets.size != 0:
                mean_perceived = np.mean(perceived_targets, axis=0)
                mean_loc[i] = np.array(((azimuth, mean_perceived[0]), (elevation, mean_perceived[1])))
                i += 1

    # divide target space in 16 half overlapping sectors and get mean response for each sector
    binned_data = np.empty((0, 4))
    bin_dict = {}
    # for a in range(0, len(azimuths) - 1, 10):
    #     for e in range(len(elevations) - 1):
    for a in range(6):
        for e in range(6):
            # select for azimuth
            tar_bin = data[np.logical_or(data[:, 0] == azimuths[a], data[:, 0] == azimuths[a + 1])]
            # print(tar_bin)
            # select for elevation
            tar_bin = tar_bin[np.logical_or(tar_bin[:, 1] == elevations[e], tar_bin[:, 1] == elevations[e + 1])]
            # print(tar_bin)
            tar_bin[:, :2] = np.array((np.mean([azimuths[a], azimuths[a + 1]]),
                                       np.mean([elevations[e], elevations[e + 1]])))
            # print(tar_bin)
            tar_bin = np.mean(tar_bin, axis=0)
            # print(tar_bin)
            # sys.exit()
            binned_data = np.vstack((binned_data, tar_bin))

    print(binned_data.shape)
    print(mean_loc.shape)



    if show:

        plt.rcParams['figure.dpi'] = 200
        if not axis:
            fig, (axis, table_axis) = plt.subplots(2, 1, height_ratios=[4, 1])
        elevation_ticks = elevations  #np.unique(targets[:, 1])
        azimuth_ticks = azimuths  #np.unique(targets[:, 0])
        # axis.set_yticks(elevation_ticks)
        # axis.set_ylim(numpy.min(elevation_ticks)-15, numpy.max(elevation_ticks)+15)
        if plot_dim == 2:
            # axis.set_xticks(azimuth_ticks)
            # axis.set_xlim(numpy.min(azimuth_ticks)-15, numpy.max(azimuth_ticks)+15)
            if show_single_responses:
                # axis.scatter(responses[:, 0], responses[:, 1], s=8, edgecolor='grey', facecolor='none')
                for azim, elev in responses:
                    # Plot thin line between response and the mean perceived location from binned_data
                    # for the perceived location get the bin it belongs to from binned_data

                    bin = binned_data[np.argmin(np.linalg.norm(binned_data[:, :2] - np.array([azim, elev]), axis=1))]
                    #Plot
                    axis.plot([azim, bin[2]], [elev, bin[3]], color='grey', linewidth=0.5, alpha=0.5)

            if binned:
                azimuths = np.unique(binned_data[:, 0])
                elevations = np.unique(binned_data[:, 1])
                mean_loc = binned_data
                # azimuth_ticks = azimuths
                # elevation_ticks = elevations
            else:
                print(mean_loc)
                mean_loc = mean_loc.reshape(mean_loc.shape[0], 4)
                print(mean_loc)
                print(mean_loc.shape)

            # Grid
            print('Here')
            for az in azimuths:  # plot lines between target locations
                print(az)
                print(mean_loc[:, 0] == az)
                print(np.where(mean_loc[:, 0] == az))
                print(mean_loc[np.where(mean_loc[:, 0] == az), 0])
                # For the current az get the indices of where they exist in mean_loc, then look up those values and put them in a list
                [x] = mean_loc[np.where(mean_loc[:, 0] == az), 0]
                [y] = mean_loc[np.where(mean_loc[:, 0] == az), 1]
                axis.plot(x, y, color='blue', linewidth=0.5, linestyle='dashed', alpha=0.5)
            for ele in elevations:
                [x] = mean_loc[np.where(mean_loc[:, 1] == ele), 0]
                [y] = mean_loc[np.where(mean_loc[:, 1] == ele), 1]
                axis.plot(x, y, color='red', linewidth=0.5, linestyle='dashed', alpha=0.5)

            # Plot mean perceived locations and lines between them and their target locations
            rainbow_colors = plt.cm.rainbow(np.linspace(0, 1, len(mean_loc)))
            for i in range(len(mean_loc)):
                color = rainbow_colors[i]
                axis.scatter(mean_loc[i, 2], mean_loc[i, 3], color=color, s=25)
                axis.plot([mean_loc[i, 0], mean_loc[i, 2]], [mean_loc[i, 1], mean_loc[i, 3]], color=color,
                          linewidth=0.5)

            for az in azimuths:  # plot lines between target locations
                [x] = mean_loc[np.where(mean_loc[:, 0] == az), 2]
                [y] = mean_loc[np.where(mean_loc[:, 0] == az), 3]
                axis.plot(x, y, color='blue', linewidth=0.5)
            for ele in elevations:
                [x] = mean_loc[np.where(mean_loc[:, 1] == ele), 2]
                [y] = mean_loc[np.where(mean_loc[:, 1] == ele), 3]
                axis.plot(x, y, color='red', linewidth=0.5)
                # Plot thin lines between single responses and mean perceived location

        elif plot_dim == 1:
            # target ids
            right_ids = np.where(data[:, 0] > 0)
            left_ids = np.where(data[:, 0] < 0)
            mid_ids = np.where(data[:, 0] == 0)
            # above_ids = numpy.where(loc_data[:, 1, 1] > 0)
            # below_ids = numpy.where(loc_data[:, 1, 1] < 0)
            axis.set_xticks(elevation_ticks)
            axis.set_xlim(np.min(elevation_ticks) - 15, np.max(elevation_ticks) + 15)
            axis.set_xlabel('target elevations')
            axis.set_ylabel('perceived elevations')
            axis.grid(visible=True, which='major', axis='both', linestyle='dashed', linewidth=0.5, color='grey')
            axis.set_axisbelow(True)
            # scatter plot with regression line (elevation gain)
            axis.scatter(targets[:, 1][left_ids], responses[:, 1][left_ids], s=10, c='red', label='left')
            axis.scatter(targets[:, 1][right_ids], responses[:, 1][right_ids], s=10, c='blue', label='right')
            axis.scatter(targets[:, 1][mid_ids], responses[:, 1][mid_ids], s=10, c='black', label='middle')
            x = np.arange(-55, 56)
            y = elevation_gain * x + n
            axis.plot(x, y, c='grey', linewidth=1, label='elevation gain %.2f' % elevation_gain)
            plt.legend()
        axis.set_yticks(elevation_ticks)
        axis.set_ylim(np.min(elevation_ticks) - 15, np.max(elevation_ticks) + 15)
        axis.set_xticks(azimuth_ticks, minor=True)
        axis.set_xlim(np.min(azimuth_ticks) - 15, np.max(azimuth_ticks) + 15)
        axis.set_xlabel('Azimuth')
        axis.set_ylabel('Elevation')
        axis.set_title('Localization accuracy')
        # Add table with results below plot, make picture bigger so table fits

        table_data = [['', 'Azimuth', 'Elevation'], ['Gain', f'-', f'{elevation_gain:.2f}'],
                      ['RMSE', f'{az_rmse:.2f}', f'{ele_rmse:.2f}'],
                      ['SD', f'{az_sd:.2f}', f'{ele_sd:.2f}'], ['Variability', f'{az_var:.2f}', f'{ele_var:.2f}']]
        table_axis.axis('off')
        table_axis.table(cellText=table_data, loc='best', cellLoc='center', colWidths=[0.1, 0.1, 0.1])

        # Make upper plot bigger by making the whole picture 2 inches higher
        fig.set_size_inches(7, 7)

        # Give more space to the plot
        plt.tight_layout()

        plt.show()

    # #  return EG, RMSE and Response Variability
    return elevation_gain, ele_rmse, ele_sd, az_rmse, az_sd


def plot_accuracy_grid(data,
                       show_single_responses=True,
                       binned=False,
                       nr_azimuth_bins=4,
                       nr_elevation_bins=4,
                       speakers_on_grid=False,
                       style=None,
                       # azimuth_min=-180,
                       # azimuth_max=175,
                       # azimuth_step=5,
                       # elevation_min=0,
                       # elevation_max=60,
                       # elevation_step=10
                       ):
    """
    better name: plot_localization_skew_grid?
    Plots the results of a localization experiment, given the data was produced with a grid of speakers.

    Problem: Hofman's original experiment used a speaker that could move freely on the x- and y-axis. Thus, the plot had
    to group multiple speaker locations onto mean points on a grid. We already use a grid of speakers. In our situation
    then, a grouping of speaker locations does not make sense, and we need a slightly different of plot that shows the
    speaker locations as a grid directly.

    Main functionality:
    X plot grid for speaker locations (input location list or grid specs; later maybe automatically extract)
    X compute mean of predictions for each speaker location and draw point there
    - bin speaker locations

    Useful:
    X rainbow colored dots (but also needs to show where which colour should be to be useful)
    X draw thin lines between each prediction point (not shown) and their mean location -> to see how far off the actual judgements were
    X Extract grid automatically (simplest: only draw grid for the azim / elev combinations that are in the data)
    X Draw thin grey lines from single predictions to their mean (or to the speaker loc...?)
    - Added info: note or result filename to be able to match plot to experiment
    - Debug mode: Add info and colors

    Basic, less important:
    - show axes in degrees w/ Azimuth and Elevation Labels, title
    - show table with elevation gain, rmse, sd, variability

    ideas not to follow right now:
    - allow median instead of mean

    :param data: a list of speaker location and predicted location pairs like [azim, elev, azim_pred, elev_pred]
    :param style: 'hofman' styles plot as in original Hofman paper, 'debug' makes it colorful
    """

    if style == 'hofman':
        speaker_grid_linestyle = 'solid'
        single_response_linewidth = 0.4
        single_response_size = 8
        single_response_color = 'black'
        show_single_responses = True
        show_single_response_lines = False
        colored_mean_pred = False
        show_mean_pred_lines = False
        pred_grid_horizontal_color = 'black'
        pred_grid_vertical_color = 'black'
        speaker_grid_color = 'black'
        show_speaker_locs = False
    elif style == 'debug':
        speaker_grid_linestyle = 'dashed'
        single_response_linewidth = 0.2
        single_response_size = 4
        single_response_color = 'grey'
        show_single_responses = False
        show_single_response_lines = True
        colored_mean_pred = True
        show_mean_pred_lines = True
        pred_grid_horizontal_color = 'blue'
        pred_grid_vertical_color = 'red'
        speaker_grid_color = 'grey'
        show_speaker_locs = True
    else:
        speaker_grid_linestyle = 'dashed'
        show_single_responses = True
        single_response_linewidth = 0.2
        single_response_size = 4
        single_response_color = 'grey'
        show_single_response_lines = True
        colored_mean_pred = False
        show_mean_pred_lines = False
        pred_grid_horizontal_color = 'black'
        pred_grid_vertical_color = 'black'
        speaker_grid_color = 'grey'
        show_speaker_locs = True


    # azimuths = [a for a in range(azimuth_min, azimuth_max + 1, azimuth_step)]
    # elevations = [e for e in range(elevation_min, elevation_max + 1, elevation_step)]

    # Get speaker locations that exist in the data -> Only draw the grid for those
    azimuths = np.unique(data[:, 0])
    elevations = np.unique(data[:, 1])

    # mean perceived location for each target speaker
    temp = []
    # Go through all azimuth and elevation combinations
    for azimuth in azimuths:
        for elevation in elevations:
            # Get all predictions for the current combination
            [perceived_targets] = data[np.where(np.logical_and(data[:, 0] == azimuth, data[:, 1] == elevation)), 2:]
            if perceived_targets.size > 0:
                # Compute their mean and save to new array
                mean_perceived = np.mean(perceived_targets, axis=0)
                temp.append((azimuth, elevation, mean_perceived[0], mean_perceived[1]))
    mean_predictions = np.stack(temp)

    # divide target space in 16 half overlapping sectors and get mean response for each sector
    # Assume data points to come from free moving speaker (like in Hofman's paper) -> no first averaging prior to sector averaging

    # For each sector: compute mean speaker location, compute mean response location
    temp = []
    min_azim = np.min(azimuths)
    max_azim = np.max(azimuths)
    min_elev = np.min(elevations)
    max_elev = np.max(elevations)
    bin_width = ((np.max(azimuths) - min_azim) / (nr_azimuth_bins + 1)) * 2
    bin_height = ((np.max(elevations) - min_elev) / (nr_elevation_bins + 1)) * 2
    for azim_bin in range(nr_azimuth_bins):
        for elev_bin in range(nr_elevation_bins):
            # Calculate borders of current sector in speaker grid
            azim_start = min_azim + (azim_bin / 2) * bin_width
            azim_end = azim_start + bin_width
            elev_start = min_elev + (elev_bin / 2) * bin_height
            elev_end = elev_start + bin_height

            # Select data points
            mask = (data[:, 0] >= azim_start) & (data[:, 0] <= azim_end) & (data[:, 1] >= elev_start) & (data[:, 1] <= elev_end)
            bin_data = np.mean(data[mask], axis=0)
            temp.append((bin_data[0], bin_data[1], bin_data[2], bin_data[3]))
    binned_data = np.stack(temp)

    bin_azimuths = np.unique(binned_data[:, 0])
    bin_elevations = np.unique(binned_data[:, 1])

    # Set up plot
    plt.rcParams['figure.dpi'] = 200
    fig, (axis, table_axis) = plt.subplots(2, 1, height_ratios=[4, 1])
    # axis.set_aspect('equal', adjustable='box')

    # Set limits (w/ margins) in advance to speed up drawing
    min_azim_pred = np.min(data[:, 2])
    max_azim_pred = np.max(data[:, 2])
    delta_azim_pred = (max_azim_pred - min_azim_pred) * 0.05
    min_elev_pred = np.min(data[:, 3])
    max_elev_pred = np.max(data[:, 3])
    delta_elev_pred = (max_elev_pred - min_elev_pred) * 0.05
    axis.set_xlim(left=min_azim_pred - delta_azim_pred, right=max_azim_pred + delta_azim_pred)
    axis.set_ylim(bottom=min_elev_pred - delta_elev_pred, top=max_elev_pred + delta_elev_pred)

    # Switch out speaker locations with sector means
    # A bit hacky, just reuses the draw method and switches out data, we need a copy of the original data to draw some details
    # add lines between speaker and sector middle
    # Change single prediction lines to point to sector mean instead of speaker mean
    if show_speaker_locs:
        axis.scatter(*np.unique(data[:, :2], axis=0).T, marker=',', s=1, linewidth=0, c='black')

    if binned:
        # Add original speaker grid as dots in the background
        # print(*np.unique(data[:, :2], axis=0).T)
        # axis.scatter(*np.unique(data[:, :2], axis=0).T, marker=',', s=1, c='grey')


        mean_predictions = binned_data
        azimuths = bin_azimuths
        elevations = bin_elevations
        show_single_response_lines = False


    def colors(azim, elev, azim_min, azim_max, elev_min, elev_max):
        r = (azim - azim_min) / (azim_max - azim_min)
        b = (elev - elev_min) / (elev_max - elev_min)
        g = 1 - (r + b) * .5
        return r, g, b

    for azim, elev, azim_pred, elev_pred in mean_predictions:
        if colored_mean_pred:
            # Color of the point where the plotted point should be, to see how garbled the predictions are
            color = colors(azim, elev, min_azim, max_azim, min_elev, max_elev)
        else:
            color = 'black'
        axis.scatter(azim_pred, elev_pred, color=color, s=15, zorder=3)
        if show_mean_pred_lines:
            # Lines between points and corresponding speaker
            axis.plot([azim, azim_pred], [elev, elev_pred], color=color, linewidth=0.5, zorder=3)

    # Plot reference color points
    axis.scatter([min_azim, min_azim, max_azim, max_azim, min_azim], [min_elev, max_elev, max_elev, min_elev, min_elev], color=[colors(azim, elev, min_azim, max_azim, min_elev, max_elev) for azim, elev in [(min_azim, min_elev), (min_azim, max_elev), (max_azim, max_elev), (max_azim, min_elev), (min_azim, min_elev)]], s=30, marker='+', zorder=0)


    # Plot speaker grid and grid of mean predictions
    for azim in azimuths:  # Vertical lines
        speaker_azims, speaker_elevs, pred_azims, pred_elevs = mean_predictions[mean_predictions[:, 0] == azim].T
        # Create boolean mask that's 1 iff the first element is our current azim, select those rows, then transpose to allow for unpacking

        # Plot vertical speaker grid lines
        axis.plot(speaker_azims, speaker_elevs, color=speaker_grid_color, linewidth=0.5, linestyle=speaker_grid_linestyle, alpha=0.5, zorder=1)

        # Plot vertical prediction grid lines
        # axis.plot(pred_azims, pred_elevs, color=pred_grid_vertical_color, linewidth=1, zorder=2)

    for elev in elevations:
        speaker_azims, speaker_elevs, pred_azims, pred_elevs = mean_predictions[mean_predictions[:, 1] == elev].T

        # Plot horizontal speaker grid lines
        axis.plot(speaker_azims, speaker_elevs, color=speaker_grid_color, linewidth=0.5, linestyle=speaker_grid_linestyle, alpha=0.5, zorder=1)

        # Plot horizontal prediction grid lines
        # axis.plot(pred_azims, pred_elevs, color=pred_grid_horizontal_color, linewidth=1, zorder=2)

    # Plot lines for single predictions
    # For each single prediction select the corresponding mean point in mean_predictions, plot a line between each single data point and it's corresponding mean aggregated point
    for single_azim, single_elev, single_azim_pred, single_elev_pred in data:
        if show_single_response_lines:
            mask = (mean_predictions[:, 0] == single_azim) & (mean_predictions[:, 1] == single_elev)
            [[mean_azim_pred, mean_elev_pred]] = mean_predictions[mask, 2:]  # Unpack possible because it must be a single value
            axis.plot([mean_azim_pred, single_azim_pred], [mean_elev_pred, single_elev_pred], color='grey', linewidth=0.5, alpha=0.3, zorder=1)
        if show_single_responses:
            # Also plot single predictions
            axis.scatter(single_azim_pred, single_elev_pred, c='None', edgecolors=single_response_color, linewidth=single_response_linewidth, s=single_response_size, zorder=1)


    plt.show()


def CNNpos_to_loc(CNN_pos, bin_size=5):
    """
    convert bin label in the CNN from Francl 2022 into [azim, elev] positions
    :param CNN_pos: int, [0, 503]
    :param bin_size: int, degree. note that elevation bin size is 2*bin_size
    :return: tuple, (azi, ele)
    """
    n_azim = int(360 / bin_size)   # bin_size=5 -> 72
    div, mod = divmod(CNN_pos, n_azim)
    azim = bin_size * mod
    if azim >= 180:
        azim -= 360
    elev = bin_size * div * 2
    if elev > 60:
        print('Elev > 60!', CNN_pos, azim, elev)

    # return elev, azim # wrong, test
    return azim, elev


def dummy_data(max_deviation=5, center_skew=0.05, nr_preds_per_speaker=2, azimuth_min=-180, azimuth_max=175,
               azimuth_step=5, elevation_min=0, elevation_max=60, elevation_step=10):
    """
    Generate dummy data for testing purposes.

    :param max_deviation: int, maximum deviation from the true location
    :param center_skew: float, skew factor for the deviation
    :param nr_preds_per_speaker: int, number of predictions per speaker
    :param azimuth_min: int, minimum azimuth angle
    :param azimuth_max: int, maximum azimuth angle
    :param azimuth_step: int, azimuth step
    :param elevation_min: int, minimum elevation angle
    :param elevation_max: int, maximum elevation angle
    :param elevation_step: int, elevation step

    :return: numpy array, (n, 4) shape, [azim, elev, azim_pred, elev_pred] columns

    """

    # Dummy data: 1000 samples, (n, 4) shape, [azim, elev, azim_pred, elev_pred] columns
    data = []
    for azim in range(azimuth_min, azimuth_max + 1, azimuth_step):
        for elev in range(elevation_min, elevation_max + 1, elevation_step):
            for _ in range(nr_preds_per_speaker):
                azim_pred = (azim + np.random.randint(-max_deviation, max_deviation + 1)) * (
                        1 - (center_skew * (abs(azim) * abs(elev)) / 10800))
                elev_pred = (elev + np.random.randint(-max_deviation, max_deviation + 1)) * (
                        1 - (center_skew * (abs(azim) * abs(elev)) / 10800))
                data.append((azim, elev, azim_pred, elev_pred))
    return np.stack(data)


def read_cnn_results(path: Path):
    """
    Given a path to a folder of results from McDermott's CNN, returns an np.array with speaker location and predicted location pairs like [azim, elev, azim_pred, elev_pred]
    For now the results of all nets are combined.

    Args:
        path: Path to a folder with .csv files containing CNN results like 'model_pred,train/azim,train/elev'

    Returns: np.array with speaker location and predicted location pairs like [azim, elev, azim_pred, elev_pred]
    """

    data = np.empty((0, 4))
    for result_file in path.iterdir():
        with open(result_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            # Turn reader into numpy array
            # Header: model_pred,train/azim,train/elev
            # Row example: 249,[29],[8]
            d = np.array([[int(row['train/azim'][1:-1]) * 5,
                           int(row['train/elev'][1:-1]) * 5,  # *2 not needed as output is in [0, 2, 4, 6, 8, 10, 12] for some reason
                           *CNNpos_to_loc(int(row['model_pred']))]
                          for row in reader])
            data = np.vstack((data, d))
    return data


def main() -> None:
    data = dummy_data(max_deviation=3, center_skew=5, nr_preds_per_speaker=3,
                   azimuth_min=-30, azimuth_max=30, azimuth_step=5,
                   elevation_min=-30, elevation_max=30, elevation_step=5)
    # data = read_cnn_results(Path('data', 'results', 'results_default'))
    # data = read_cnn_results(Path('data', 'results', 'results_test_2024-11-25'))
    print(np.unique(data[:, :2], axis=0))
    print(f'All data shape: {data.shape}')

    plot_accuracy_grid(data, nr_elevation_bins=5, nr_azimuth_bins=5, binned=False, show_single_responses=True, style='debug')


if __name__ == "__main__":
    main()
