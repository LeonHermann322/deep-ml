import numpy as np

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	return np.linalg.eig(matrix)[0].real
