import numpy as np
import pandas as pd


def sliding_windows(data, window_size=7, step_size=1):
    # This function returns the sliding windows and the labels for each window
    # If output_3d is True, the output will be a 3D array, otherwise it will be a 2D array
    # The 2D array is (num_samples, num_features), useful for ML and the 3D array is (num_samples, window_size/time_steps, num_features), useful for RNNs
    # The difference is that the window is not flattened in the 3D array

    X = []
    y = []

    for i in range(0, len(data) - window_size + 1, step_size):
        end = i + window_size
        if end > len(data):
            end = len(data)

        window = data[i:end]
        if len(window) < window_size:
            # Padding the window with zeros and the same amount of columns
            padding = np.zeros(
                (window_size - len(window), len(window.columns)))
            # Stacking the window and padding keeping the same order
            window = np.vstack([window, padding])

        # Get the label from the first row of the window
        label = window.iloc[0]['label'] if 'label' in window.columns else ValueError(
            "Data must contain a 'label' column for classification.")
        window = window.drop(
            columns=['label'])

        X.append(np.array(window).flatten())
        y.append(label)

    return np.array(X), np.array(y)


def one_hot_encode(column_vector, one_hot_keys):
    # This function takes a column vector and a dictionary of keys to one-hot encode the column
    # The keys are the unique values in the column and the values are the one-hot encoded values
    # The output is a 2D array with the one-hot encoded values
    encoded_array = np.zeros(len(column_vector))

    # Iterate over the column vector and encode the values
    for i, behavior in enumerate(column_vector):
        if behavior in one_hot_keys:
            encoded_array[i] = one_hot_keys[behavior]
        else:
            raise ValueError(f"Value {i} not found in one_hot_keys")

    return encoded_array


def fill_synthetic_data(merged_data, percentage):

    # This function fills the real data with synthetic data based on the percentage
    # The real data is a DataFrame and the synthetic data is a DataFrame
    # The output is a DataFrame with the same columns as the real data and the synthetic data
    # The output is a DataFrame with the same number of rows as the real data
    real_data = merged_data[merged_data['origin'] == 'real']
    synth_data_normal = merged_data[(merged_data['origin'] == 'synth') & (
        merged_data['label'] == 'normal')]
    synth_data_agg = merged_data[(merged_data['origin'] == 'synth') & (
        merged_data['label'] == 'aggressive')]

    n_synth_samples = int(len(real_data) * percentage)
    synth_data = pd.concat(
        [synth_data_normal.iloc[:n_synth_samples//2], synth_data_agg.iloc[:n_synth_samples//2]], ignore_index=True)

    return pd.concat([real_data, synth_data], ignore_index=True).drop(columns=['origin'])
