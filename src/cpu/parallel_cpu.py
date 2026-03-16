import numpy as np
import multiprocessing as mp

def _compute_chunk(args):
    """
    Worker function for multiprocessing.

    Parameters
    ----------
    args : tuple
        (chunk, Z_T, T)

    Returns
    -------
    np.ndarray
        Partial correlation matrix block.
    """
    chunk, Z_T, T = args
    return (chunk @ Z_T) / (T - 1)


def parallel_cpu_correlation(X, num_workers=4):
    """
    Compute correlation matrix using multiprocessing.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (N, T)
    num_workers : int
        Number of CPU processes

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

    Z_T = Z.T

    # Split rows into chunks
    chunks = np.array_split(Z, num_workers)

    # Prepare arguments
    tasks = [(chunk, Z_T, T) for chunk in chunks]

    # Multiprocessing pool
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_compute_chunk, tasks)

    # Combine results
    C = np.vstack(results)

    return C

