
from collections import Counter

def confusion_matrix(data):
	# Implement the function here
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for true, pred in data:
		if true == 1:
			if pred == 1:
				tp += 1
			else:
				fn += 1
		else:
			if pred == 1:
				fp += 1
			else:
				tn += 1

	return [[tp, fn],[fp, tn]]
