import numpy as np

def phi_transform(data: list[float], degree: int) -> list[list[float]]:
	"""
	Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

	Args:
		data (list[float]): A list of numerical values to transform.
		degree (int): The degree of the polynomial expansion.

	"""
	if degree < 0:
		return []

	# degree shape n
	result = []
	for i in range(len(data)):
		row = []
		for j in range(0,degree+1):
			row.append(float(np.power(data[i], j)))
		result.append(row)

	# target shape n x degree
	return result