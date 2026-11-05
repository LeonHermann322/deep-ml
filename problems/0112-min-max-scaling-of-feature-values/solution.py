def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    min_x = min(x)
    max_x = max(x)
    return [(el - min_x) / (max_x - min_x) for el in x]