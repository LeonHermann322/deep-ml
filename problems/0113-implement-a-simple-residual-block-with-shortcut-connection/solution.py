import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	out1 = w1 @ x
	activ1 = np.where(out1 < 0, 0, out1)
	out2 = w2 @ activ1
	activ2 = np.where(out2 < 0, 0, out2)
	residual = activ2 + x
	return np.where(residual < 0, 0, residual)