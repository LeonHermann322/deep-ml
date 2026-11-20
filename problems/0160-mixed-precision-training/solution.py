import numpy as np

class MixedPrecision:
    def __init__(self, loss_scale=1024.0):
        # Initialize loss scaling factor
        self.loss_scale = loss_scale
    
    def forward(self, weights, inputs, targets):
        # Perform forward pass with float16, return scaled loss as a Python float
        weights, inputs, targets = weights.astype(np.float16), inputs.astype(np.float16), targets.astype(np.float16)
        preds = inputs @ weights
        diffs = targets - preds
        unscaled_loss = float(np.mean(np.square(diffs)))
        return self.loss_scale * unscaled_loss

    def backward(self, gradients):
        # Unscale gradients and check for overflow, return as float32
        gradients = gradients.astype(np.float32)
        overflow_exists = np.isnan(gradients).sum() > 0 or np.isinf(gradients).sum() > 0
        if overflow_exists:
            return np.zeros_like(gradients)

        return gradients / self.loss_scale