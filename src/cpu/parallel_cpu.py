import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory


# -------------------------------------------------------
# BASELINE (your original)
# -------------------------------------------------------

def _compute_chunk_baseline(args):
    shm_name, shape, dtype, start_idx, end_idx, T = args

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    Z = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    chunk = Z[start_idx:end_idx]
    result = (chunk @ Z.T) / (T - 1)

    existing_shm.close()
    return result


def parallel_cpu_correlation_baseline(X, num_workers=4):
    N, T = X.shape

    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    Z = (X - means) / stds

    shm = shared_memory.SharedMemory(create=True, size=Z.nbytes)
    shared_Z = np.ndarray(Z.shape, dtype=Z.dtype, buffer=shm.buf)
    np.copyto(shared_Z, Z)

    chunk_size = int(np.ceil(N / num_workers))
    tasks = []

    for i in range(num_workers):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, N)
        if start < N:
            tasks.append((shm.name, Z.shape, Z.dtype, start, end, T))

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_compute_chunk_baseline, tasks)

    C = np.vstack(results)

    shm.close()
    shm.unlink()

    return C


# -------------------------------------------------------
# OPTIMIZED
# -------------------------------------------------------

def _compute_chunk_optimized(args):
    shm_name, shape, dtype, start_idx, end_idx, T = args

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    Z = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    N = shape[0]
    result = np.zeros((end_idx - start_idx, N), dtype=dtype)

    for local_i, i in enumerate(range(start_idx, end_idx)):
        Zi = Z[i]

        for j in range(i, N):  # upper triangle only
            result[local_i, j] = np.dot(Zi, Z[j]) / (T - 1)

        result[local_i, i] = 1.0

    existing_shm.close()
    return (start_idx, result)


def parallel_cpu_correlation_optimized(X, num_workers=4):
    N, T = X.shape

    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    Z = (X - means) / stds

    shm = shared_memory.SharedMemory(create=True, size=Z.nbytes)
    shared_Z = np.ndarray(Z.shape, dtype=Z.dtype, buffer=shm.buf)
    np.copyto(shared_Z, Z)

    chunk_size = int(np.ceil(N / num_workers))
    tasks = []

    for i in range(num_workers):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, N)
        if start < N:
            tasks.append((shm.name, Z.shape, Z.dtype, start, end, T))

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_compute_chunk_optimized, tasks)

    C = np.zeros((N, N), dtype=Z.dtype)

    for start_idx, block in results:
        rows = block.shape[0]
        C[start_idx:start_idx + rows] = block

    # Mirror lower triangle
    i_lower = np.tril_indices(N, -1)
    C[i_lower] = C.T[i_lower]

    shm.close()
    shm.unlink()

    return C

