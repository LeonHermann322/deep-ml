from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    best_loss = val_losses[0]
    best_epoch = 0
    counter = 0
    for i in range(1, len(val_losses)):
        if val_losses[i] < best_loss - min_delta:
            # Meaningful improvement
            best_loss = val_losses[i]
            best_epoch = i
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                return (i, best_epoch)

    return (len(val_losses) - 1, best_epoch)