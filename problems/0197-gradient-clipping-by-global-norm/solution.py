import numpy as np

def clip_gradients_by_global_norm(gradients: list[list[float]], max_norm: float) -> list[list[float]]:
    """
    Clip gradients by global norm.
    
    Args:
        gradients: List of gradient arrays
        max_norm: Maximum allowed global norm
    
    Returns:
        List of clipped gradient arrays
    """
    # Your code here
    norm = np.sqrt(np.sum([np.sum(np.square(np.asarray(gradient))) for gradient in gradients]))

    if norm <= max_norm:
        return gradients

    scale_factor = max_norm / norm

    return [np.asarray(gradient) * scale_factor for gradient in gradients]

    