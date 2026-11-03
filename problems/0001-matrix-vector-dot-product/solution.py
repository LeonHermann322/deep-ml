def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	dim1 = len(a)

	if dim1 == 0:
		return -1

	dim2 = len(a[0])

	if dim2 != len(b):
		return -1

	return_val = []
	for row in range(dim1):
		dot_prod_row = sum([a[row][i] * b[i] for i in range(dim2)])
		return_val.append(dot_prod_row)

	return return_val