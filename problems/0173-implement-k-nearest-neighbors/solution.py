import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    def calc_euclidean(a, b):
        diffs = np.asarray(a) - np.asarray(b)
        return np.sqrt(np.sum(np.square(diffs)))

    distances = {i: calc_euclidean(point, query_point) for i, point in enumerate(points)}
    nearest = sorted(distances, key=distances.get)[:k]
    return [points[i] for i in nearest]
    