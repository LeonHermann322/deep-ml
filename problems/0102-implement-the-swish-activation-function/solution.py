import math

def swish(x: float) -> float:
	"""
	Implements the Swish activation function.

	Args:
		x: Input value

	Returns:
		The Swish activation value
	"""
	sigmoid = 1 / (1 + math.exp(-x))
	return x * sigmoid