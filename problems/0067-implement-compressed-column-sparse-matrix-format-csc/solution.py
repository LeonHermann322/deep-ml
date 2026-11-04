import numpy as np
def compressed_col_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

	:param dense_matrix: List of lists representing the dense matrix
	:return: Tuple of (values, row indices, column pointer)
	"""
	dense_matrix = np.asarray(dense_matrix)
	row_idx, col_idx = dense_matrix.nonzero()
	order = np.argsort(col_idx)
	values = dense_matrix[row_idx[order], col_idx[order]]
	col_ptr = np.concatenate(([0],np.cumsum(np.bincount(col_idx, minlength=dense_matrix.shape[1]))))
	return values.tolist(), row_idx[order].tolist(), col_ptr.tolist()
