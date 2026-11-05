import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    if len(p) != len(q) or len(p) == 0:
        return 0.0
    return -float(np.log(np.sum(np.sqrt(np.asarray([p[i] * q[i] for i in range(len(p))])))))