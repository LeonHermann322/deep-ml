
import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	classes = set(y)
	samples = len(y)
	y = np.array(y)

	probabilities = [np.sum(y == class_idx) / samples for class_idx in classes]
	val = 1 - np.sum([np.square(probability) for probability in probabilities])

	return round(val,3)