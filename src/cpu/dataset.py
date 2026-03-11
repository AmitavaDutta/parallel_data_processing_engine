import numpy as np

def generate_dataset(N, T, seed=0):
    """
    Generate a synthetic dataset of N parallel time series, each with T time steps.

    Parameters
    ----------
    N : int
        Number of parallel time series (rows in the dataset).
    T : int
        Number of time steps per time series (columns in the dataset).
    seed : int, optional (default=0)
        Random seed for reproducibility of the generated dataset.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (N, T) containing synthetic time series data
        sampled from a standard normal distribution (mean=0, std=1).
    
    Notes
    -----
    This dataset can be used for testing correlation matrix computation and
    benchmarking CPU/GPU implementations.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate random dataset: each row is a time series
    data = np.random.randn(N, T)
    
    return data

