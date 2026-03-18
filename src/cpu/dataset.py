import numpy as np
import pandas as pd

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


def read_dataset(file_path):
    """
    Read a dataset from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing the dataset. The file should have N rows
        and T columns, where N is the number of time series and T is the number
        of time steps.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (N, T) containing the dataset read from the CSV file.
    
    Notes
    -----
    The CSV file should not contain headers or index columns. Each row should represent
    a time series, and each column should represent a time step.
    """
    # Read the dataset with a header and index column
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    #print(data)
    # remove the first column which is the date
    #data = data[:, 1:]
    #print(data.head())
    data_transposed = data.T
    return data_transposed


#read_dataset("../global_temperature_comparison_modified.csv")

