import numpy as np

def compute_correlation_serial(X):
    """
    Compute the full correlation matrix of a dataset of time series using a single-threaded CPU approach.

    Parameters
    ----------
    X : np.ndarray
        A 2D NumPy array of shape (N, T), where N is the number of time series
        and T is the number of time steps per series.

    Returns
    -------
    np.ndarray
        The N x N correlation matrix, where element (i, j) is the Pearson correlation
        coefficient between time series i and j.

    Notes
    -----
    This implementation is single-threaded (serial) and serves as a baseline for 
    comparing multi-core CPU and GPU-accelerated versions.
    """
    N, T = X.shape  # Number of time series and number of time steps

    # Compute mean and standard deviation for each time series (row-wise)
    means = X.mean(axis=1, keepdims=True)   # Shape (N, 1)
    stds = X.std(axis=1, keepdims=True)     # Shape (N, 1)

    # Standardize each time series (Z-score normalization)
    Z = (X - means) / stds

    # Compute the correlation matrix: C = Z * Z^T / (T - 1)
    C = (Z @ Z.T) / (T - 1)

    return C

