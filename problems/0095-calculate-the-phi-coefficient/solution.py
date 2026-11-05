import numpy as np
import math

def phi_corr(x: list[int], y: list[int]) -> float:
	"""
	Calculate the Phi coefficient between two binary variables.

	Args:
	x (list[int]): A list of binary values (0 or 1).
	y (list[int]): A list of binary values (0 or 1).

	Returns:
	float: The Phi coefficient rounded to 4 decimal places.
	"""
	x = np.asarray(x).astype(bool)
	y = np.asarray(y).astype(bool)

	x00 = float((~x & ~y).sum())
	x01 = float((~x & y).sum())
	x10 = float((x & ~y).sum())
	x11 = float((x & y).sum())

	val = (x00 * x11 - x01 * x10) / math.sqrt((x00+x01)*(x10+x11)*(x00+x10)*(x01+x11)) if ((x00+x01)*(x10+x11)*(x00+x10)*(x01+x11)) != 0 else 0

	return round(val,4)