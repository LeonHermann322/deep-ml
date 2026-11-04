import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """

    percent_25 = np.percentile(data, 25)
    percent_75 = np.percentile(data, 75)

    unique_counts = np.unique_counts(data)
    max_idx = np.argmax(unique_counts.counts)
    max_el = unique_counts.values[max_idx]

    return {
        "mean" : float(np.mean(data)),
        "median" : float(np.median(data)),
        "mode": float(max_el),
        "variance" : float(np.var(data)),
        "standard_deviation" : float(np.std(data)),
        "25th_percentile":float(percent_25 ),
        "50th_percentile": float(np.percentile(data, 50)),
        "75th_percentile": float(percent_75),
        "interquartile_range": float(percent_75 - percent_25)
    }
    