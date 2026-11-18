import numpy as np

def compute_confusion_matrix(y_true, y_pred, num_classes, normalize=None, round_decimals=4):
    """
    Compute a KxK confusion matrix with optional normalization.

    Args:
        y_true: Iterable of true labels in [0, K-1]
        y_pred: Iterable of predicted labels in [0, K-1]
        num_classes: K, number of classes
        normalize: None | 'true' | 'pred' | 'all'
        round_decimals: decimals to round when normalization is applied

    Returns:
        list[list[int|float]] confusion matrix
    """
    confusion = np.zeros([num_classes, num_classes])
    for i in range(len(y_true)):
        target = y_true[i]
        prediction = y_pred[i]
        confusion[target, prediction] += 1

    if normalize == 'true':
        true_class_count = np.asarray([max(np.sum(np.asarray(y_true) == clss), 1) for clss in range(num_classes)])
        confusion = np.transpose(np.transpose(confusion) / true_class_count)
    elif normalize == 'pred':
        pred_class_count = np.asarray([max(np.sum(np.asarray(y_pred) == clss), 1) for clss in range(num_classes)])
        confusion /= pred_class_count
    elif normalize == 'all':
        confusion /= len(y_pred)

    return confusion.round(round_decimals)