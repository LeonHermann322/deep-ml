import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    X = np.asarray(X).tolist()
    if y is not None:
        y = np.asarray(y).tolist()

    batches = []
    for offset in range(0, len(X), batch_size):
        batchX = X[offset:offset + batch_size]

        if y is not None:
            batchY = y[offset:offset + batch_size]
            batches.append([batchX, batchY])
        else:
            batches.append(batchX)

    return batches