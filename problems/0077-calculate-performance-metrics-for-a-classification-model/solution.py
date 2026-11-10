import numpy as np


def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    actual = np.asarray(actual).astype(bool)
    predicted = np.asarray(predicted).astype(bool)

    tp = np.sum(actual & predicted)
    tn = np.sum(~actual & ~predicted)
    fp = np.sum(~actual & predicted)
    fn = np.sum(actual & ~predicted)

    confusion_matrix = [[tp, fn], [fp, tn]]

    accuracy = (tp + tn) / actual.shape[0]
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    negativePredictive = tn / (tn + fn)
    f1 = 2 * recall * precision / (precision + recall)

    return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
