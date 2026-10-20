import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	nd_arr = np.array(matrix)

	if mode == "column":
		nd_arr = np.transpose(nd_arr)

	means = []
	for i in range(nd_arr.shape[0]):
		mean = np.mean(nd_arr[i])
		means.append(mean)

	return means

	