import numpy as np

def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
    
    Returns:
        Recall value as a float
    """
    true_bool = np.asarray(y_true).astype(bool)
    pred_bool = np.asarray(y_pred).astype(bool)

    tp = np.sum(true_bool & pred_bool)
    fn = np.sum(true_bool & ~pred_bool)

    return tp / (tp + fn) if (tp + fn) != 0 else 0
