
import numpy as np

def r_squared(y_true, y_pred):
	# Write your code here
	sse = np.sum(np.square(y_true - y_pred))
	mean_true = np.mean(y_true)
	sst = np.sum(np.square(y_true - mean_true))
	return 1 - sse / sst