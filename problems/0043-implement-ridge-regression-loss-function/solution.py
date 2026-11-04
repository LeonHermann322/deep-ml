import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	predictions = np.dot(X, w)
	diffs = y_true - predictions
	squared = np.square(diffs)
	mse = np.mean(squared)
	ridge = alpha * np.sum(np.square(w))
	return mse + ridge
