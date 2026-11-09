import numpy as np

def calculate_correlation_matrix(X, Y=None):
	if Y is None:
		return np.corrcoef(X.T)

    X = np.asarray(X)
    Y = np.asarray(Y)

    return np.corrcoef(X.T, Y.T)[:X.shape[1], X.shape[1]:]