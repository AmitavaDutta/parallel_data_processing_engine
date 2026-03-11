import numpy as np
from concurrent.futures import ThreadPoolExecutor

def parallel_cpu_correlation(X, num_workers=4):
    """
    Compute the full correlation matrix of a dataset of time series using multi-threaded CPU parallelism.

    Parameters
    ----------
    X : np.ndarray
        A 2D NumPy array of shape (N, T), where N is the number of time series
        and T is the number of time steps per series.
    num_workers : int, optional (default=4)
        Number of worker threads to use for parallel computation.

    Returns
    -------
    np.ndarray
        The N x N correlation matrix, where element (i, j) is the Pearson correlation
        coefficient between time series i and j.

    Notes
    -----
    - This implementation splits the standardized time series matrix Z into chunks,
      each handled by a separate thread.
    - Each thread computes a portion of the correlation matrix (chunk @ Z.T) / (T - 1).
    - Results are vertically stacked to form the full correlation matrix.
    - Due to Python's GIL, using ThreadPoolExecutor may not always give maximum
      CPU speedup for NumPy operations; multiprocessing could be explored for
      further performance gains.
    """
    N, T = X.shape  # Number of time series and number of time steps

    # Compute mean and standard deviation for each time series (row-wise)
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)

    # Standardize each time series (Z-score normalization)
    Z = (X - means) / stds

    # Define function to compute a chunk of the correlation matrix
    def compute_chunk(chunk):
        # chunk shape: (N_chunk, T)
        # Z.T shape: (T, N)
        # Result shape: (N_chunk, N)
        return (chunk @ Z.T) / (T - 1)

    # Split Z into approximately equal chunks along the row axis
    chunks = np.array_split(Z, num_workers)

    # Compute each chunk in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(compute_chunk, chunks))

    # Combine all chunk results into the full correlation matrix
    C = np.vstack(results)

    return C

