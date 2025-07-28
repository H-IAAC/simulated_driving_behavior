import os
import pandas as pd
import numpy as np
from itertools import product


def get_data(directory: str, driver: str, specifier: str, sensor: str) -> pd.DataFrame:
    """
    Load UAH data from the specified directory for a given driver and segment.
    Args:
        directory (str): Path to the directory containing the data.
        driver (str): Driver identifier.
        specifier (str): Segment specifier (e.g., 'NORMAL1-SECONDARY').
        sensor (str): Sensor type ('acc' for accelerometer, 'gps' for GPS).
    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    if sensor == 'acc':
        for i in os.listdir(os.path.join(directory, driver)):
            if i.endswith(specifier):
                data = pd.read_csv(os.path.join(directory, driver, i, 'RAW_ACCELEROMETERS.txt'), sep=' ', header=None, names=[
                    'timestamp',
                    'system_active',
                    'acc_x',
                    'acc_y',
                    'acc_z',
                    'acc_x_KF',
                    'acc_y_KF',
                    'acc_z_KF',
                    'Roll',
                    'Pitch',
                    'Yaw'
                ], usecols=range(11))
                data = data.drop(['system_active'], axis=1)
                data['acc'] = np.sqrt(
                    data['acc_x']**2 + data['acc_y']**2 + data['acc_z']**2)

                return data

    elif sensor == 'gps':
        for i in os.listdir(os.path.join(directory, driver)):
            if i.endswith(specifier):
                data = pd.read_csv(os.path.join(directory, driver, i, 'RAW_GPS.txt'), sep=' ', header=None, names=[
                    'timestamp',
                    'speed',
                    'lat',
                    'lon',
                    'altitude',
                    'vert_accuracy',
                    'horiz_accuracy',
                    'course',
                    'difcourse',
                ], usecols=range(9))

                return data


def load_and_correct(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load new data and correct timestamps to ensure continuity with existing data.
    Args:
        existing_df (pd.DataFrame): Existing DataFrame to which new data will be added.
        new_df (pd.DataFrame): New DataFrame to be added.
    Returns:
        pd.DataFrame: Combined DataFrame with corrected timestamps.
    """
    if existing_df.empty:
        return new_df
    last_ts = existing_df['timestamp'].iloc[-1]
    new_df = new_df.copy()
    new_df['timestamp'] += last_ts

    return pd.concat([existing_df, new_df], axis=0)


def read_data(drivers: list, directory: str) -> dict:
    """
    Load and correct the data for all drivers and sensors.
    Args:
        drivers (list): List of driver identifiers.
        directory (str): Path to the directory containing the data.
    Returns:
        dict: A dictionary containing the accelerometer and GPS data for normal and aggressive driving segments.
    """

    # Function to load and correct timestamps in order to keep them continuous
    for sensor in ['acc', 'gps']:
        normal = pd.DataFrame()
        aggressive = pd.DataFrame()

        for driver in drivers:
            # Normal driving segments
            for segment in ['NORMAL1-SECONDARY', 'NORMAL2-SECONDARY', 'NORMAL-MOTORWAY']:
                new_data = get_data(directory, driver, segment, sensor=sensor)
                normal = load_and_correct(normal, new_data)

            # Aggressive driving segments
            for segment in ['AGGRESSIVE-SECONDARY', 'AGGRESSIVE-MOTORWAY']:
                new_data = get_data(directory, driver, segment, sensor=sensor)
                aggressive = load_and_correct(aggressive, new_data)

        if sensor == 'acc':
            df_acc = {'normal': normal, 'aggressive': aggressive}
        else:
            df_gps = {'normal': normal, 'aggressive': aggressive}

    return {'acc': df_acc, 'gps': df_gps}


def get_samples_per_second(df_acc: pd.DataFrame) -> float:
    """
    Calculate the samples per second from the accelerometer data.
    Args:
        df_acc (pd.DataFrame): DataFrame containing accelerometer data with a 'timestamp' column.
    Returns:
        float: The number of samples per second.
    """
    # Assuming the timestamp is in seconds and the data is sorted by timestamp
    samples_per_second = df_acc['normal'].iloc[1]['timestamp'] - \
        df_acc['normal'].iloc[0]['timestamp']
    df_acc['normal'].iloc[0]['timestamp']
    samples_per_second = 1 / samples_per_second

    return samples_per_second


def load_synthetic_data(town_data_directory: str) -> tuple:
    """Load synthetic data from the specified town data directory.
    Args:
        town_data_directory (str): Path to the directory containing the town data.

    Returns:
        tuple: Two dictionaries containing CARLA and SUMO data for fixed and LLM modes with traffic conditions.
    """

    carla_path = os.path.join(town_data_directory, 'carla')
    sumo_path = os.path.join(town_data_directory, 'sumo')

    carla_data = {}
    carla_data['fixed'] = {}
    carla_data['fixed']['no_traffic'] = {}
    carla_data['fixed']['traffic'] = {}
    carla_data['llm'] = {}
    carla_data['llm']['no_traffic'] = {}
    carla_data['llm']['traffic'] = {}

    sumo_data = {}
    sumo_data['fixed'] = {}
    sumo_data['fixed']['no_traffic'] = {}
    sumo_data['fixed']['traffic'] = {}
    sumo_data['llm'] = {}
    sumo_data['llm']['no_traffic'] = {}
    sumo_data['llm']['traffic'] = {}

    for sim, mode, traffic in product(['carla', 'sumo'], ['fixed', 'llm'], ['no_traffic', 'traffic']):

        if sim == 'carla':
            path = os.path.join(carla_path, mode, traffic)
        else:
            path = os.path.join(sumo_path, mode, traffic)

        if os.path.exists(path):
            files = os.listdir(path)

            for beh in ['normal', 'aggressive']:
                for file in files:
                    if file.endswith(f'{beh}.csv'):
                        csv = pd.read_csv(
                            os.path.join(path, file), sep=',', header=0)

                        if sim == 'carla':
                            # If the behavior already exists, concatenate the data
                            if beh in carla_data[mode][traffic]:
                                carla_data[mode][traffic][beh] = pd.concat(
                                    [carla_data[mode][traffic][beh], csv], axis=0, ignore_index=True)
                            else:
                                carla_data[mode][traffic][beh] = csv

                        else:
                            # If the behavior already exists, concatenate the data
                            if beh in sumo_data[mode][traffic]:
                                sumo_data[mode][traffic][beh] = pd.concat(
                                    [sumo_data[mode][traffic][beh], csv], axis=0, ignore_index=True)
                            else:
                                sumo_data[mode][traffic][beh] = csv

        else:
            print(f"Path {path} does not exist. Skipping...")

    return carla_data, sumo_data


def stack_data(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, percentage: float = 0.5) -> pd.DataFrame:
    """Merge real and synthetic data based on a specified percentage.

    Args:
        real_data (dict): Dictionary containing real data.
        synthetic_data (dict): Dictionary containing synthetic data.
        percentage (float): Percentage of synthetic data keeping the same size as real data.

    Returns:
        dict: Merged dataset.
    """
    n_synth_samples = int(percentage * len(real_data))
    n_real_samples = int((1 - percentage) * len(real_data))
    if n_synth_samples > len(synthetic_data):
        print(
            f"Warning: Requested synthetic size {n_synth_samples} exceeds available synthetic data size {len(synthetic_data)}. Using all available synthetic data.")
        n_synth_samples = len(synthetic_data)

    print(
        f"Using {n_synth_samples} synthetic samples to merge with real data {len(real_data)} real samples.")
    print(
        f"Percentage of synthetic data: {(n_synth_samples /  (n_synth_samples + n_real_samples)) * 100:.2f}%")

    # Select the first n_synth_samples from the synthetic data
    synthetic_data = synthetic_data.iloc[:n_synth_samples]
    real_data = real_data.iloc[:n_real_samples]

    # Concatenate real and synthetic data
    merged_data = pd.concat([real_data, synthetic_data],
                            axis=0, ignore_index=True)

    return merged_data


def load_merged_data(merged_path: str) -> tuple:
    """Load merged datasets from the specified path.
    Args:
        merged_path (str): The path to the merged datasets.
    Returns:
        tuple: A tuple containing the merged SUMO and CARLA datasets.
    """
    merged_sumo_uah = {
        'fixed': {
            'normal': pd.read_csv(f'{merged_path}/sumo_uah/fixed/merged_sumo_uah_normal.csv'),
            'aggressive': pd.read_csv(f'{merged_path}/sumo_uah/fixed/merged_sumo_uah_aggressive.csv')
        },
        'llm': {
            'normal': pd.read_csv(f'{merged_path}/sumo_uah/llm/merged_sumo_uah_normal.csv'),
            'aggressive': pd.read_csv(f'{merged_path}/sumo_uah/llm/merged_sumo_uah_aggressive.csv')
        }
    }
    merge_carla_uah = {
        'fixed': {
            'normal': pd.read_csv(f'{merged_path}/carla_uah/fixed/merged_carla_uah_normal.csv'),
            'aggressive': pd.read_csv(f'{merged_path}/carla_uah/fixed/merged_carla_uah_aggressive.csv')
        },
        'llm': {
            'normal': pd.read_csv(f'{merged_path}/carla_uah/llm/merged_carla_uah_llm_normal.csv'),
            'aggressive': pd.read_csv(f'{merged_path}/carla_uah/llm/merged_carla_uah_llm_aggressive.csv')
        }
    }
    return merged_sumo_uah, merge_carla_uah
