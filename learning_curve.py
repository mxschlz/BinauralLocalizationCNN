import os
import glob
import json
import numpy as np
import pandas as pd


def get_learning_curve_json(root):
    """
    Extracts learning curve data from JSON files in a directory and returns it as a Pandas DataFrame.

    Parameters:
    - root (str): The root directory containing subdirectories with JSON files.

    Returns:
    - data (pd.DataFrame): A DataFrame with columns "iteration" and "accuracy" containing the learning curve data.
    """

    # List subdirectories within the specified root directory
    children = sorted(os.listdir(root))
    index = list(zip(*children))
    multi_index = pd.MultiIndex.from_tuples(children)

    # Create two DataFrames to store iteration and accuracy data
    data_iterations = pd.DataFrame(columns=["iteration"])
    data_accuracy = pd.DataFrame(columns=["accuracy"])

    # Loop through subdirectories and process JSON files
    for d in children:
        full_path = os.path.join(root, d)

        # Find JSON files within the subdirectory
        file = glob.glob(f"{full_path}/*json")

        if file:
            # Load JSON content and extract iterations and accuracy values
            content = np.array(json.load(open(file[0])))
            iterations = content[:, 0]
            accuracy = content[:, 1]

            # Concatenate the extracted data into respective DataFrames
            data_iterations = pd.concat([data_iterations, pd.DataFrame({"iteration": iterations})], ignore_index=True)
            data_accuracy = pd.concat([data_accuracy, pd.DataFrame({"accuracy": accuracy})], ignore_index=True)
        else:
            # Skip directories without JSON files
            pass

    # Combine the two DataFrames into one, resulting in a learning curve DataFrame
    data = pd.concat([data_iterations, data_accuracy], axis=1)

    return data


if __name__ =="__main__":
    data = get_learning_curve_json(root="netweights_MSL")
