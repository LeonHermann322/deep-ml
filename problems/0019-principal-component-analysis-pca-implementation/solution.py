import numpy as np


import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    # Standardize
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    data = (data - means) / stds

    # Covariance matrix
    cov = np.cov(data.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort eigenvectors by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Select top k components
    components = eigenvectors[:, :k]

    # Fix signs: first non-zero element must be positive
    for i in range(k):
        nonzero = np.flatnonzero(components[:, i])

        if len(nonzero) > 0 and components[nonzero[0], i] < 0:
            components[:, i] *= -1

    return np.round(components, 4)

