import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	features_np = np.array(features)
	weights_np = np.array(weights)
	labels_np = np.array(labels)

	raw_logits = features_np @ weights_np + bias
	probabilities = 1 / (1 + np.exp(-raw_logits))
	mse = np.mean(np.power(probabilities - labels_np, 2))
	
	return probabilities, mse