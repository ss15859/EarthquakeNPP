import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

def find_mainshock_times(earthquake_catalog, magnitude_threshold):
    """
    Find the mainshock times for earthquakes above a given magnitude threshold.

    Parameters:
        earthquake_catalog (pandas.DataFrame): DataFrame containing earthquake catalog.
        magnitude_threshold (float): Magnitude threshold for mainshock selection.

    Returns:
        list: List of mainshock times.
    """
    # Filter earthquakes above the magnitude threshold
    above_threshold = earthquake_catalog[earthquake_catalog['magnitude'] >= magnitude_threshold]
    
    # Get unique mainshock times
    mainshock_times = above_threshold['time'].unique().tolist()
    
    return mainshock_times

def calculate_expected_threshold(time_difference, mainshock_magnitude):
    """
    Calculate the expected magnitude threshold based on time difference from the mainshock.

    Parameters:
        time_difference (array-like): Time difference between each event and the mainshock (in days).
        mainshock_magnitude (float): Magnitude of the mainshock.

    Returns:
        array-like: Expected magnitude threshold for each event.
    """

    threshold = mainshock_magnitude/2 - 0.25 - np.log10(time_difference)
    threshold[time_difference<=0] = 0
    return threshold

def apply_detection_threshold(earthquake_catalog, mainshock_times):
    """
    Apply detection threshold to aftershocks of each mainshock.

    Parameters:
        earthquake_catalog (pandas.DataFrame): DataFrame containing earthquake catalog.
        mainshock_times (list): List of mainshock times (in days).

    Returns:
        pandas.DataFrame: Modified earthquake catalog after applying detection threshold.
    """

    for mainshock_time in mainshock_times:
        # Get mainshock magnitude
        mainshock_magnitude = earthquake_catalog.loc[earthquake_catalog['time'] == mainshock_time, 'magnitude'].iloc[0]
        
        # Calculate time difference between each event and the mainshock (in days)
        time_difference = (earthquake_catalog['time'] - mainshock_time).dt.total_seconds() / (24 * 60 * 60)  # converting seconds to days
        
        # Calculate expected magnitude threshold for each event
        expected_threshold = calculate_expected_threshold(time_difference, mainshock_magnitude)
        
        # Filter events below the expected magnitude threshold for the mainshock
        catalog_mainshock_removed = earthquake_catalog[earthquake_catalog['magnitude'] >= expected_threshold]

        # print("removed events: ", len(earthquake_catalog)- len(catalog_mainshock_removed))
        
        # Concatenate with filtered catalog
        earthquake_catalog = catalog_mainshock_removed
    
    return earthquake_catalog

if __name__ == '__main__':

    with open("simulate_ETAS_California_catalog_config.json", 'r') as f:
        config = json.load(f)

    catalog = pd.read_csv(
                    config["fn_store"],
                    index_col=0,
                    parse_dates=["time"],
                    dtype={"url": str, "alert": str},
                )

    catalog = catalog.sort_values(by='time')

    # Set the magnitude threshold for mainshock selection
    magnitude_threshold = 5.2

    # Find the mainshock times
    mainshock_times = find_mainshock_times(catalog, magnitude_threshold)

    print('Number of Mainshocks:', len(mainshock_times))

    # Apply detection threshold to aftershocks of each mainshock
    filtered_catalog = apply_detection_threshold(catalog, mainshock_times)

    print("Original number of events:", len(catalog))
    print("Number of events after applying time and magnitude dependent detection threshold:", len(filtered_catalog))


    filtered_catalog[["latitude", "longitude", "time", "magnitude"]].to_csv("ETAS_California_incomplete_catalog.csv")
