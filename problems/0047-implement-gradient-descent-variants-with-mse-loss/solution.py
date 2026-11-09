import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    def param_update(sample_input, sample_target, w):
        # input: (batch_s, n_features)
        # target: (batch_s,)
        # weights: (n_features)
        input_size = sample_input.shape[0]
        pred = sample_input @ w
        diffs = pred - sample_target
        grad = 2 * np.mean(diffs[:, np.newaxis] * sample_input, axis=0)
        return w - learning_rate * grad

    for epoch in range(n_epochs):
        if method == 'batch':
            weights = param_update(X,y, weights)
        elif method == 'stochastic':
            for i in range(X.shape[0]):
                weights = param_update(X[i][np.newaxis, :], y[i][np.newaxis], weights)
        else:
            batch_count = X.shape[0] // batch_size
            for i in range(batch_count):
                batch_inp, batch_target = X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]
                weights = param_update(batch_inp, batch_target, weights)

    return weights
