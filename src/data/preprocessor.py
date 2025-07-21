import numpy as np


def sliding_windows(data, label, window_size=7, step_size=1):
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

        X.append(np.array(window).flatten())
        y.append(label)

    return np.array(X), np.array(y)
