import numpy as np
import os


# -------------------------------------------------------
# BASELINE (your original)
# -------------------------------------------------------

def compute_correlation_blockwise_baseline(X, block_size=1000, use_memmap=False):
    N, T = X.shape

    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    Z = (X - means) / stds

    if use_memmap:
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

            block = (Zi @ Zj.T) / (T - 1)
            C[i:i_end, j:j_end] = block

    if use_memmap:
        C.flush()
        return C

    return C


# -------------------------------------------------------
# OPTIMIZED
# -------------------------------------------------------

def compute_correlation_blockwise_optimized(X, block_size=1000, use_memmap=False):
    N, T = X.shape

    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    Z = (X - means) / stds

    if use_memmap:
        os.makedirs("results", exist_ok=True)
        filename = f"results/temp_corr_{os.getpid()}.dat"
        C = np.memmap(filename, dtype='float64', mode='w+', shape=(N, N))
    else:
        C = np.zeros((N, N), dtype=Z.dtype)

    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        Zi = Z[i:i_end]

        for j in range(i, N, block_size):  # upper triangle only
            j_end = min(j + block_size, N)
            Zj = Z[j:j_end]

            block = (Zi @ Zj.T) / (T - 1)

            C[i:i_end, j:j_end] = block

            if i != j:
                C[j:j_end, i:i_end] = block.T

    if use_memmap:
        C.flush()
        return C

    return C

