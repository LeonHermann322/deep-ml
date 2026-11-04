import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

	:param dense_matrix: 2D list representing a dense matrix
	:return: A tuple containing (values array, column indices array, row pointer array)
	"""
	dense_matrix = np.asarray(dense_matrix)

	non_zero = dense_matrix.nonzero()
	
	values = dense_matrix[non_zero]
	column_indices = non_zero[1]

	row_ptr = np.zeros(dense_matrix.shape[0] + 1, dtype=int)

	row_ptr[1:] = np.cumsum(np.bincount(non_zero[0], minlength=dense_matrix.shape[0]))

	return values.tolist(), column_indices.tolist(), row_ptr.tolist()