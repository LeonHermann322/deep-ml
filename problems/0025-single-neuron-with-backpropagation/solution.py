import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	weights = initial_weights
	bias = initial_bias
	losses = []

	for epoch in range(epochs):
		# Forward
		predictions = 1 / (1+np.exp(-(features @ weights + bias)))

		# Loss
		diffs = predictions - labels
		mse = np.mean(np.square(diffs))
		losses.append(mse)

		# Backward
		bias_grad = 2*np.mean(diffs * predictions * (1 - predictions))
		weight_grads = 2*np.stack([np.mean(diffs * predictions * (1 - predictions) * features[:,j]) for j in range(len(weights))])

        # Apply gradient
		bias += -bias_grad * learning_rate
		weights += -weight_grads * learning_rate


	return weights.round(4).tolist(), bias.round(4), [loss.round(4) for loss in losses]