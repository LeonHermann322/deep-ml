import numpy as np

def make_diagonal(x):
	N = len(x)
	standard_diag = np.identity(N)
	return standard_diag * x
