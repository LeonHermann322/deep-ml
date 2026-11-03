import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here

	means = np.mean(data, axis=0)
	stds = np.std(data, axis=0)
	mins = np.min(data, axis=0)
	maxs = np.max(data, axis=0)

	return (data - means) / stds, (data - mins) / (maxs - mins)