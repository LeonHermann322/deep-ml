import numpy as np

def mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error between two arrays.

    Parameters:
        y_true (numpy.ndarray): Array of true values
        y_pred (numpy.ndarray): Array of predicted values

    Returns:
        float: Mean Absolute Error
    """
    diffs = y_true - y_pred
    absolute_diffs = np.abs(diffs)
    return np.mean(absolute_diffs)