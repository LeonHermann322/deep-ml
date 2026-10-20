def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	if len(a) == 0:
		return -1


	columns_a = len(a[0])

	if columns_a != len(b):
		return -1

	dot_prod = []
	for row in a:
		summed = 0
		for i in range(columns_a):
			summed += row[i] * b[i]
		dot_prod.append(summed)
	
	return dot_prod