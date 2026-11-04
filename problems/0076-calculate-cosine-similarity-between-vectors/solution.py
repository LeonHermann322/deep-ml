import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	dot = np.dot(v1, v2)
	len1 = np.sqrt(np.sum(np.square(v1)))
	len2 = np.sqrt(np.sum(np.square(v2)))
	return dot / (len1 * len2)