import os
import glob
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


def get_learning_curve_json(root, type="auc"):
    """
    Extracts learning curve data from JSON files in a directory and returns it as a Pandas DataFrame
    with a multi-index.

    Parameters:
    - root (str): The root directory containing subdirectories with JSON files.

    Returns:
    - data (pd.DataFrame): A DataFrame with multi-index ('network', 'iteration') and 'accuracy' columns.
    """

    # List subdirectories within the specified root directory
    children = sorted(os.listdir(root))

    # Create an empty list to store dataframes for each network
    network_dfs = []

    # Loop through subdirectories and process JSON files
    for d in children:
        full_path = os.path.join(root, d)

        # Find JSON files within the subdirectory
        files = glob.glob(f"{full_path}/*{type}*.json")

        if files:
            # Load JSON content and extract iterations and accuracy values
            content = np.array(json.load(open(files[0])))
            iterations = content[:, 0]
            accuracy = content[:, 1]

            # Create a DataFrame for the current network
            network_df = pd.DataFrame({'iteration': iterations, 'accuracy': accuracy})

            # Add the network name as a new column in the DataFrame
            network_df['network'] = d

            # Append the DataFrame to the list
            network_dfs.append(network_df)

    # Concatenate all network DataFrames into one
    data = pd.concat(network_dfs)

    # Set the multi-index
    data.set_index(['network'], inplace=True)

    return data


fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
data = get_learning_curve_json(root="netweights_MSL")
sns.lineplot(x="iteration", y="accuracy", data=data, ax=ax[0])
sns.lineplot(x="iteration", y="accuracy", data=data, hue="network", ax=ax[1])
plt.show()
