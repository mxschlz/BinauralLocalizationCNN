import numpy as np


_p_ref = 2e-5  # 20 μPa, the standard reference pressure for sound in air


def spectemp_coverage(sound_composition, dyn_range, upper_freq):
    """
    Calculate the spectro-temporal coverage of a sound composition.

    Parameters:
    - sound_composition (list): A list of sounds in the composition.
    - dyn_range (float): The dynamic range in decibels (dB) to consider for coverage calculation.
    - upper_freq (int): The upper frequency limit to consider for the analysis.

    Returns:
    - float: The ratio of the number of spectro-temporal points within the specified dynamic range
             to the total number of spectro-temporal points in the composition.

    Raises:
    - ValueError: If the input sound_composition is not a list.

    Note:
    - This function assumes that the sounds in sound_composition have a spectrogram() method that
      returns the spectrogram of the sound.

    """
    assert isinstance(sound_composition, list), ValueError("Input must be a list of sounds")

    # Combine the sounds in the composition
    sound = sum(sound_composition)

    # Calculate the spectrogram and power
    freqs, times, power = sound.spectrogram(show=False)
    if upper_freq:
        power = power[freqs < upper_freq, :]

    # Convert power to logarithmic scale for plotting
    power = 10 * np.log10(power / (_p_ref ** 2))

    # Calculate the maximum and minimum dB values
    dB_max = power.max()
    dB_min = dB_max - dyn_range

    # Select the interval of power values within the specified dynamic range
    interval = power[np.where((power > dB_min) & (power < dB_max))]

    # Calculate the ratio of points within the dynamic range to total points
    coverage = interval.shape[0] / power.flatten().shape[0]

    return coverage


def get_azimuth_from_df(dataset):
    """
    Extracts azimuth values from a dataset and returns them as a list.

    Parameters:
    - dataset (iterable): The dataset from which to extract azimuth values.

    Returns:
    - list: A list of azimuth values extracted from the dataset.

    """

    azimuth = list()  # List to store the extracted azimuth values

    # Iterate over each element in the dataset
    for x in dataset:
        if x is None:
            azimuth.append(None)  # Append None if the element is None
        else:
            azimuth.append(x[0])  # Append the first element of x to the azimuth list

    return azimuth


def get_elevation_from_df(dataset):
    """
    Extracts elevation values from a dataset and returns them as a list.

    Parameters:
    - dataset (iterable): The dataset from which to extract elevation values.

    Returns:
    - list: A list of elevation values extracted from the dataset.

    """

    elevation = list()  # List to store the extracted elevation values

    # Iterate over each element in the dataset
    for x in dataset:
        if x is None:
            elevation.append(None)  # Append None if the element is None
        else:
            elevation.append(x[1])  # Append the second element of x to the elevation list

    return elevation


def replace_in_array(array, to_replace_val=None, replace_with_val=0):
    """
    Replaces values in an array with a specified replacement value.

    Parameters:
    - array (list or ndarray): The array in which values will be replaced.
    - to_replace_val (object, optional): The value to be replaced. Default is None.
    - replace_with_val (object, optional): The replacement value. Default is 0.

    Returns:
    - list or ndarray: An array with the specified values replaced.

    """

    for i, val in enumerate(array):
        if val == to_replace_val:
            array[i] = replace_with_val  # Replace the value at index i with replace_with_val

    return array
