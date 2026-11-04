
import numpy as np

def jaccard_index(y_true, y_pred):
	# Write your code here
	true = np.asarray(y_true).astype(bool)
	pred = np.asarray(y_pred).astype(bool)
	return round(np.sum(true & pred) / np.sum(true | pred), 3)
