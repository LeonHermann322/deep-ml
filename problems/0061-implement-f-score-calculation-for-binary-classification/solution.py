import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	true_bool = np.asarray(y_true).astype(bool)
	pred_bool = np.asarray(y_pred).astype(bool)

	tp = np.sum(true_bool & pred_bool)
	fn = np.sum(true_bool & ~pred_bool)
	fp = np.sum(~true_bool & pred_bool)

	recall = tp / (tp + fn) if (tp + fn) != 0 else 0
	precision = tp / (tp + fp) if (tp + fp) != 0 else 0

	beta **= 2

	return round(((1 + beta) * recall * precision / (recall + beta * precision)) if (recall + beta * precision) != 0 else 0, 3)
