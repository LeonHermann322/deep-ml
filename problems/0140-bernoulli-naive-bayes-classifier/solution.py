import numpy as np


class NaiveBayes:
    def __init__(self, smoothing=1.0):
        # Laplace/Lidstone smoothing strength (1.0 = classic Laplace)
        self.smoothing = smoothing

    def fit(self, X, y):
        # Fit model to binary features X (n_samples, n_features) and labels y
        X = np.asarray(X).astype(int)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        pos_mask = y == 1
        neg_mask = ~pos_mask
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()

        # Class priors (in log space)
        self.log_prior_pos = np.log(n_pos / n_samples)
        self.log_prior_neg = np.log(n_neg / n_samples)

        # P(feature = value | class), shape (n_features, 2), with smoothing
        pos_probs = np.zeros((n_features, 2))
        neg_probs = np.zeros((n_features, 2))

        for feature in range(n_features):
            for value in [0, 1]:
                x_mask = X[:, feature] == value
                pos_count = np.sum(x_mask & pos_mask)
                neg_count = np.sum(x_mask & neg_mask)

                # Denominator adds smoothing * (number of possible values = 2)
                pos_probs[feature, value] = (pos_count + self.smoothing) / (n_pos + 2 * self.smoothing)
                neg_probs[feature, value] = (neg_count + self.smoothing) / (n_neg + 2 * self.smoothing)

        # Store log-probabilities to avoid numerical underflow when multiplying
        self.pos_log_probs = np.log(pos_probs)
        self.neg_log_probs = np.log(neg_probs)
        return self

    # Keep the original method name working
    forward = fit

    def predict(self, X):
        # Predict class labels (1.0 / 0.0) for test set X
        X = np.asarray(X).astype(int)
        n_features = X.shape[1]
        feature_idx = np.arange(n_features)[None, :]  # broadcasts against X

        # Look up log P(x_j | class) for every sample/feature, then sum over features
        pos_scores = self.log_prior_pos + self.pos_log_probs[feature_idx, X].sum(axis=1)
        neg_scores = self.log_prior_neg + self.neg_log_probs[feature_idx, X].sum(axis=1)

        return (pos_scores > neg_scores).astype(float)