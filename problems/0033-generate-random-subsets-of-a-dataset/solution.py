import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    n = X.shape[0]
    subsets = []
    for i in range(n_subsets):
        subset = np.random.choice(n, size=n if replacements else n // 2, replace=replacements)
        input_subset = X[subset]
        output_subset = y[subset]
        subsets.append((input_subset, output_subset))
    return subsets