import numpy as np
from typing import List, Tuple
from itertools import chain

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    fold_size = n_samples // k
    rest = n_samples % k

    indices = list(range(n_samples))

    if shuffle:
        np.random.shuffle(indices)

    folds = []
    for fold_idx in range(k):
        start_idx = fold_idx * fold_size + (rest if fold_idx != 0 else 0)
        end_idx = start_idx + fold_size + (rest if fold_idx == 0 else 0)

        folds.append(indices[start_idx:end_idx])

    cv_folds = []
    for fold_idx in range(k):
        test_indices = folds[fold_idx]
        train_indices = list(chain.from_iterable([fold for i, fold in enumerate(folds) if i != fold_idx]))
        cv_folds.append((train_indices, test_indices))

    return cv_folds

