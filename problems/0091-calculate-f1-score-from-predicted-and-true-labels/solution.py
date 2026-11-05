import numpy as np

def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	true = np.asarray(y_true).astype(bool)
	pred = np.asarray(y_pred).astype(bool)

	tp = np.sum(true & pred)
	tn = np.sum(~true & ~pred)
	fp = np.sum(~true & pred)
	fn = np.sum(true & ~pred)

	recall = tp / (tp + fn) if (tp + fn) != 0 else 0
	precision = tp / (tp + fp) if (tp + fp) != 0 else 0

	f1 = 2 * recall * precision / (recall + precision) if (recall + precision) != 0 else 0

	return round(f1,3)