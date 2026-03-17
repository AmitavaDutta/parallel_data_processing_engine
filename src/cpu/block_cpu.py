import numpy as np
import os

def compute_correlation_blockwise(X, block_size=1000, use_memmap=False):
    """
    Compute correlation matrix using block-wise multiplication.
    Optionally uses memory-mapping for very large N.
    """
    N, T = X.shape

    # Standardize
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    Z = (X - means) / stds

    if use_memmap:
        # Create a temporary file for the matrix
        filename = "results/temp_corr.dat"
        C = np.memmap(filename, dtype='float64', mode='w+', shape=(N, N))
    else:
        C = np.zeros((N, N))

    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        Zi = Z[i:i_end]

        for j in range(0, N, block_size):
            j_end = min(j + block_size, N)
            Zj = Z[j:j_end]

            # Standard Pearson correlation block computation
            block = (Zi @ Zj.T) / (T - 1)
            C[i:i_end, j:j_end] = block
    
    if use_memmap:
        # Flush changes to disk
        C.flush()
        # Note: In a real scenario, you would keep the memmap or return it.
        # For benchmarking, we return the array view.
        return C
    
    return C
