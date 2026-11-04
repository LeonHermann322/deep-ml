import numpy as np
def calculate_brightness(img):
	# Write your code here
	if len(img) == 0:
		return -1
	if len({len(row) for row in img}) != 1:
		return -1
	np_img = np.asarray(img)

	if np.sum((np_img < 0) | (np_img > 255)) > 0:
		return -1

	return round(np.mean(np_img), 2)

