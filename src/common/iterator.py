import numpy as np

def data_iterator(X, y, batch_size, shuffle=True):
    index = list(range(len(X)))
    if shuffle:
        np.random.shuffle(index)

    for start_idx in range(0, len(X), batch_size):
        end_idx = min(start_idx + batch_size, len(X))
        yield X[index[start_idx: end_idx]], y[index[start_idx: end_idx]]