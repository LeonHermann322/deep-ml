import numpy as np
def precision(y_true, y_pred):
	# TP / (TP + FP)
	bool_true = np.asarray(y_true).astype(bool)
	bool_pred = np.asarray(y_pred).astype(bool)

	tp = np.sum(bool_true & bool_pred)
	fp = np.sum(~bool_true & bool_pred)

	return tp / (tp + fp) if (tp + fp) != 0 else 0
