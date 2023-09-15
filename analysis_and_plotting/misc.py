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
