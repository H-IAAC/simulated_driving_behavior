import numpy as np
import pandas as pd


def sliding_windows(data: pd.DataFrame, window_size: int = 7, step_size: int = 1) -> tuple:
    """    
    This function generates sliding windows from the input data.
    The data should be a pandas DataFrame with a 'label' column for classification tasks.
    Args:
        data (pd.DataFrame): Input data containing features and a 'label' column.
        window_size (int): Size of the sliding window.
        step_size (int): Step size for the sliding window.
    Returns:
        X (np.ndarray): 2D array of shape (num_samples, num_features)
        y (np.ndarray): 1D array of labels corresponding to each window.
    """
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


def one_hot_encode(column_vector: np.ndarray, one_hot_keys: dict) -> np.ndarray:
    """
    This function one-hot encodes a column vector based on the provided keys.
    The keys are expected to be a dictionary where the keys are the unique values in the column
    and the values are the one-hot encoded values.
    Args:
        column_vector (pd.Series or np.ndarray): The column vector to be one-hot encoded.
        one_hot_keys (dict): A dictionary mapping unique values to one-hot encoded values.
    Returns:
        np.ndarray: A 2D array with the one-hot encoded values.
    """
    encoded_array = np.zeros(len(column_vector))

    # Iterate over the column vector and encode the values
    for i, behavior in enumerate(column_vector):
        if behavior in one_hot_keys:
            encoded_array[i] = one_hot_keys[behavior]
        else:
            raise ValueError(f"Value {i} not found in one_hot_keys")

    return encoded_array


def fill_synthetic_data(merged_data: pd.DataFrame, percentage: float) -> pd.DataFrame:
    """
    This function fills the real data with synthetic data based on the percentage.
    The real data is a DataFrame and the synthetic data is a DataFrame.
    The output is a DataFrame with the same columns as the real data and the synthetic data
    The output is a DataFrame with the same number of rows as the real data.
    Args:
        merged_data (pd.DataFrame): DataFrame containing both real and synthetic data.
        percentage (float): Percentage of synthetic data to be added to the real data.
    Returns:
        pd.DataFrame: DataFrame containing the real data and the synthetic data combined.
    """

    real_data = merged_data[merged_data['origin'] == 'real']
    synth_data_normal = merged_data[(merged_data['origin'] == 'synth') & (
        merged_data['label'] == 'normal')]
    synth_data_agg = merged_data[(merged_data['origin'] == 'synth') & (
        merged_data['label'] == 'aggressive')]

    n_synth_samples = int(len(real_data) * percentage)
    synth_data = pd.concat(
        [synth_data_normal.iloc[:n_synth_samples//2], synth_data_agg.iloc[:n_synth_samples//2]], ignore_index=True)

    return pd.concat([real_data, synth_data], ignore_index=True).drop(columns=['origin'])
