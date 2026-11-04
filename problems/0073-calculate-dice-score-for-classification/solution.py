
import numpy as np

def dice_score(y_true, y_pred):
	true = np.asarray(y_true).astype(bool)
	pred = np.asarray(y_pred).astype(bool)

	tp = np.sum(true & pred)
	fp = np.sum(~true & pred)
	fn = np.sum(true & ~pred)

	divisor = 2*tp + fp + fn

	return round(2*tp / divisor if divisor != 0 else 0, 3)
