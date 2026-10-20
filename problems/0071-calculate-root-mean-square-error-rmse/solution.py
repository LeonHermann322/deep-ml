
import numpy as np

def rmse(y_true, y_pred):
	diffs = y_true - y_pred
	squared = np.power(diffs, 2)
	mean = np.mean(squared)
	rmse_res = np.sqrt(mean)


	return round(rmse_res,3)
