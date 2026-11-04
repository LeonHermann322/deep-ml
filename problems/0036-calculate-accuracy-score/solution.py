import numpy as np

def accuracy_score(y_true, y_pred):
	prediction_count = len(y_true)
	bool_y_true = np.asarray(y_true).astype(bool)
	bool_y_pred = np.asarray(y_pred).astype(bool)

	total_correct = (bool_y_pred == bool_y_true).sum()
	return total_correct / prediction_count