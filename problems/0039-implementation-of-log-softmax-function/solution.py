import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	max_score = max(scores)
	shifted_scores = [score - max_score for score in scores]
	sum_scores = sum([np.exp(shifted_score) for shifted_score in shifted_scores])

	return [shifted_score - np.log(sum_scores) for shifted_score in shifted_scores]