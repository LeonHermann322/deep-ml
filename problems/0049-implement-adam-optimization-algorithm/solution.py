import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only — the objective value `f` is never evaluated.
    # Your code here
    params = x0
    m = np.zeros_like(params)
    v = np.zeros_like(params)

    for i in range(1, num_iterations+1):
        sample_grad = grad(params)
        m = beta1 * m + (1 - beta1) * sample_grad
        v = beta2 * v + (1 - beta2) * np.square(sample_grad)
        m_update = m / (1 - beta1 ** i)
        v_update = v / (1 - beta2 ** i)
        params -= learning_rate * m_update / (np.sqrt(v_update) + epsilon)
    return params