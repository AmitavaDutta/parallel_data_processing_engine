import numpy as np


def compute_correlation_blockwise(X, block_size=1000):
    """
    Compute correlation matrix using block-wise multiplication.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, T)
    block_size : int
        Number of time series per block

    Returns
    -------
    np.ndarray
        Correlation matrix (N x N)
    """

    N, T = X.shape

    # Standardize
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    Z = (X - means) / stds

    C = np.zeros((N, N))

    for i in range(0, N, block_size):

        i_end = min(i + block_size, N)
        Zi = Z[i:i_end]

        for j in range(0, N, block_size):

            j_end = min(j + block_size, N)
            Zj = Z[j:j_end]

            block = (Zi @ Zj.T) / (T - 1)

            C[i:i_end, j:j_end] = block

    return C

