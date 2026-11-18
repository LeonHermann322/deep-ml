import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute the Huber Loss between true and predicted values.

    Args:
        y_true (float | list[float]): Ground truth values
        y_pred (float | list[float]): Predicted values
        delta (float): Transition threshold between MSE and MAE behavior

    Returns:
        float: Average Huber loss
    """
    res = np.asarray(y_true) - np.asarray(y_pred)

    loss = np.where(np.abs(res) <= delta, np.square(res) / 2, delta * (np.abs(res) - delta / 2))

    return loss.mean()
    