import math

def softmax(scores: list[float]) -> list[float]:
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    sum_scores = sum(exp_scores)
    softmax_scores = [exp_score / sum_scores for exp_score in exp_scores] 
    return softmax_scores