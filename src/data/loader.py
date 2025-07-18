import os
import pandas as pd
import numpy as np


def get_data(directory, driver, specifier, sensor):

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


def read_accelerometer(drivers, directory):
    normal = pd.DataFrame()
    aggressive = pd.DataFrame()
    drowsy = pd.DataFrame()
    for driver in drivers:
        normal = pd.concat(
            [normal, get_data(directory, driver, 'NORMAL1-SECONDARY', sensor='acc')], axis=0)
        normal = pd.concat(
            [normal, get_data(directory, driver, 'NORMAL2-SECONDARY', sensor='acc')], axis=0)
        normal = pd.concat(
            [normal, get_data(directory, driver, 'NORMAL-MOTORWAY', sensor='acc')], axis=0)
        aggressive = pd.concat([aggressive, get_data(
            directory, driver, 'AGGRESSIVE-SECONDARY', sensor='acc')], axis=0)
        aggressive = pd.concat([aggressive, get_data(
            directory, driver, 'AGGRESSIVE-MOTORWAY', sensor='acc')], axis=0)
        drowsy = pd.concat(
            [drowsy, get_data(directory, driver, 'DROWSY-SECONDARY', sensor='acc')], axis=0)
        drowsy = pd.concat(
            [drowsy, get_data(directory, driver, 'DROWSY-MOTORWAY', sensor='acc')], axis=0)

    df_accelerometer = {}
    df_accelerometer['normal'] = normal
    df_accelerometer['aggressive'] = aggressive
    df_accelerometer['drowsy'] = drowsy

    return df_accelerometer


def read_gps(drivers, directory):
    normal = pd.DataFrame()
    aggressive = pd.DataFrame()
    drowsy = pd.DataFrame()
    for driver in drivers:
        normal = pd.concat(
            [normal, get_data(directory, driver, 'NORMAL1-SECONDARY', sensor='gps')], axis=0)
        normal = pd.concat(
            [normal, get_data(directory, driver, 'NORMAL2-SECONDARY', sensor='gps')], axis=0)
        normal = pd.concat(
            [normal, get_data(directory, driver, 'NORMAL-MOTORWAY', sensor='gps')], axis=0)
        aggressive = pd.concat([aggressive, get_data(
            directory, driver, 'AGGRESSIVE-SECONDARY', sensor='gps')], axis=0)
        aggressive = pd.concat([aggressive, get_data(
            directory, driver, 'AGGRESSIVE-MOTORWAY', sensor='gps')], axis=0)
        drowsy = pd.concat(
            [drowsy, get_data(directory, driver, 'DROWSY-SECONDARY', sensor='gps')], axis=0)
        drowsy = pd.concat(
            [drowsy, get_data(directory, driver, 'DROWSY-MOTORWAY', sensor='gps')], axis=0)

    df_gps = {}
    df_gps['normal'] = normal
    df_gps['aggressive'] = aggressive
    df_gps['drowsy'] = drowsy

    return df_gps


def get_samples_per_second(df_acc):
    """Calculate the samples per second from the accelerometer data."""
    # Assuming the timestamp is in seconds and the data is sorted by timestamp
    samples_per_second = df_acc['normal'].iloc[1]['timestamp'] - \
        df_acc['normal'].iloc[0]['timestamp']
    samples_per_second = 1 / samples_per_second
    return samples_per_second


def load_synthetic_data(town_data_directory):
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

    for file in os.listdir(os.path.join(sumo_path, 'fixed')):
        sumo_data['fixed']['normal']
