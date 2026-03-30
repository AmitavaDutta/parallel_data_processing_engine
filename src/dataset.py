import numpy as np
import pandas as pd
import os

# -----------------------------
# Synthetic Dataset Generator
# -----------------------------
def generate_dataset(N, T, seed=0):
    """
    Generate a synthetic dataset of N parallel time series, each with T time steps.
    Returns: np.ndarray of shape (N, T)
    """
    np.random.seed(seed)
    return np.random.randn(N, T)


# -----------------------------
# Real Dataset Reader
# -----------------------------
def read_dataset(file_path):
    """
    Read real dataset from CSV.
    Returns: np.ndarray of shape (N, T), where N = time series, T = time steps.

    Prescribed format:
        - CSV file, each column is a time series
        - No Date column needed
        - All numeric
    """
    if not os.path.exists(file_path):
        raise ValueError(f"Failed to open file: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {file_path}") from e

    # Ensure numeric only
    if not all(np.issubdtype(dtype, np.number) for dtype in df.dtypes):
        raise ValueError(
            "All columns must be numeric. "
            "Prescribed format is in data_format_readme.md"
        )

    # Convert to numpy array
    data = df.values

    # Shape: (N x T)
    return data.T  # transpose so each row is a time series
