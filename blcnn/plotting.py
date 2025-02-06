import csv
import glob
from pathlib import Path

import keras
import numpy as np
import visualkeras
from PIL import ImageFont
from matplotlib import pyplot as plt

from blcnn.util import load_config


def main() -> None:
    # Go through labels in config and create one plot for each label and net

    plotting_config = load_config('blcnn/config.yml').plotting
    for hrtf_label in plotting_config.hrtf_labels:
        print(f'Plotting for HRTF: {hrtf_label}')
        # Load data available in the folder 'data/output/{hrtf_label}'
        data_folder = Path(f'data/output/{hrtf_label}')
        for result_file in glob.glob(str(data_folder / '*.csv')):
            print(f'Generating plot for file: {result_file}')
            data = read_single_cnn_result(Path(result_file))
            title = f'Localization Accuracy: {hrtf_label} - {Path(result_file).stem}'
            plt = plot_localization_accuracy(data,
                                             nr_elevation_bins=plotting_config.nr_elevation_bins,
                                             nr_azimuth_bins=plotting_config.nr_azimuth_bins,
                                             binned=plotting_config.binned,
                                             show_single_responses=plotting_config.show_single_responses,
                                             style=plotting_config.style)
            plt.title(title)
            plt.savefig(data_folder / f'{Path(result_file).stem}.png', dpi=400)

            # Clear the plot for the next file
            plt.clf()
            plt.close()


    # data = dummy_data(max_deviation=3, center_skew=2, nr_preds_per_speaker=3,
    #                azimuth_min=-30, azimuth_max=30, azimuth_step=5,
    #                elevation_min=-30, elevation_max=30, elevation_step=5)
    # plot_localization_accuracy(data, nr_elevation_bins=7, nr_azimuth_bins=9, binned=True, show_single_responses=False, style='debug')


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
            d = np.array([[*CNNpos_to_loc(int(row['true_class'])),
                           *CNNpos_to_loc(int(row['pred_class']))]
                          for row in reader])
            # d = np.array([[int(row['train/azim'][1:-1]) * 5,
            #                int(row['train/elev'][1:-1]) * 5,  # *2 not needed as output is in [0, 2, 4, 6, 8, 10, 12] for some reason
            #                *CNNpos_to_loc(int(row['model_pred']))]
            #               for row in reader])
            data = np.vstack((data, d))
    return data

def read_single_cnn_result(path: Path):
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return np.array([[*CNNpos_to_loc(int(row['true_class'])),
                          *CNNpos_to_loc(int(row['pred_class']))]
                         for row in reader])


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
    # Fold back to front
    if azim > 90:
        azim = 180 - azim
    elif azim < -90:
        azim = -180 - azim

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


def plot_localization_accuracy(data,
                               show_single_responses=True,
                               binned=False,
                               nr_azimuth_bins=4,
                               nr_elevation_bins=4,
                               speakers_on_grid=False,
                               style=None,
                               ) -> plt:
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
            # -> If certain coords don't have an example, the speaker grid won't be square
            temp.append((bin_data[0], bin_data[1], bin_data[2], bin_data[3]))
    binned_data = np.stack(temp)

    bin_azimuths = np.unique(binned_data[:, 0])
    bin_elevations = np.unique(binned_data[:, 1])

    # Set up plot
    plt.rcParams['figure.dpi'] = 400
    # fig, (axis, table_axis) = plt.subplots(2, 1, height_ratios=[4, 1])
    axis = plt.subplot()
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
        # Doesn't work in binned mode if there are target locations missing as the mean bin locations are not equal anymore for all azims and elevs in a row or column

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
        # break
        if show_single_response_lines:
            # For correct color: get coords of the target speaker location that the corresponding mean point belongs to
            # For correct position: get coords of the single response and coords of the mean point it belongs to
            mask = (mean_predictions[:, 0] == single_azim) & (mean_predictions[:, 1] == single_elev)
            [[mean_azim, mean_elev, mean_azim_pred, mean_elev_pred]] = mean_predictions[mask, :]  # Unpack possible because it must be a single value
            axis.plot([mean_azim_pred, single_azim_pred], [mean_elev_pred, single_elev_pred], color=colors(mean_azim, mean_elev, min_azim, max_azim, min_elev, max_elev), linestyle='--', linewidth=0.2, alpha=0.3, zorder=1)
        if show_single_responses:
            # Also plot single predictions
            axis.scatter(single_azim_pred, single_elev_pred, c='None', edgecolors=single_response_color, linewidth=single_response_linewidth, s=single_response_size, zorder=1)


    # plt.show()
    # plt.savefig('test.png', dpi=400)
    return plt


def plot_model_diagram(model: keras.Sequential):
    def _text_callable(layer_index, layer):
        # Every other piece of text is drawn above the layer, the first one below
        above = bool(layer_index % 2)

        # Get the output shape of the layer
        output_shape = [x for x in list(layer.output_shape) if x is not None]

        # If the output shape is a list of tuples, we only take the first one
        if isinstance(output_shape[0], tuple):
            output_shape = list(output_shape[0])
            output_shape = [x for x in output_shape if x is not None]

        # Variable to store text which will be drawn
        output_shape_txt = ""

        layer_cfg = layer.get_config()
        if 'conv2d' in layer_cfg['name']:
            output_shape_txt += f"Conv2D"
            output_shape_txt += f"\nKernel size:\n{layer_cfg['kernel_size'][0]}x{layer_cfg['kernel_size'][1]}\nStrides:\n{layer_cfg['strides'][0]}x{layer_cfg['strides'][1]}"
        elif 'max_pooling2d' in layer_cfg['name']:
            output_shape_txt += f"Pool"
            output_shape_txt += f"\nPool size:\n{layer_cfg['pool_size'][0]}x{layer_cfg['pool_size'][0]}\nStrides:\n{layer_cfg['strides'][0]}x{layer_cfg['strides'][1]}"
        elif 'dense' in layer_cfg['name']:
            output_shape_txt += f"Dense"
            output_shape_txt += f"\n{layer_cfg['units']} units"
        elif 'dropout' in layer_cfg['name']:
            output_shape_txt += f"Dropout"
        elif 'batch_normalization' in layer_cfg['name']:
            output_shape_txt += f"BN"
        elif 'relu' in layer_cfg['name']:
            output_shape_txt += f"ReLU"

        # Create a string representation of the output shape
        output_shape_txt += "\nOutput shape:\n"
        for ii in range(len(output_shape)):
            output_shape_txt += str(output_shape[ii])
            if ii < len(output_shape) - 2:  # Add an x between dimensions, e.g. 3x3
                output_shape_txt += "x"
            if ii == len(output_shape) - 2:  # Add a newline between the last two dimensions, e.g. 3x3 \n 64
                output_shape_txt += "\n"

        # Return the text value and if it should be drawn above the layer
        return output_shape_txt, above

    img = visualkeras.layered_view(model, legend=True, text_callable=_text_callable,
                                   font=ImageFont.load_default(size=36), scale_xy=0.5, min_xy=10, scale_z=0.5)
    plt.imshow(np.asarray(img))
    plt.show()


if __name__ == "__main__":
    main()
