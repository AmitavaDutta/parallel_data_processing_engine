import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory

def _compute_chunk(args):
    """
    Worker function for multiprocessing using shared memory.
    """
    shm_name, shape, dtype, start_idx, end_idx, T = args
    
    # Attach to the existing shared memory block
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    
    # Reconstruct the shared NumPy array
    Z = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    
    # Extract the specific row chunk for this worker to compute
    chunk = Z[start_idx:end_idx]
    
    # Compute partial correlation block
    result = (chunk @ Z.T) / (T - 1)
    
    # Clean up local reference to shared memory
    existing_shm.close()
    
    return result


def parallel_cpu_correlation(X, num_workers=4):
    """
    Compute correlation matrix using multiprocessing and shared memory.
    """
    N, T = X.shape

    # Standardize
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    Z = (X - means) / stds

    # Create shared memory block for Z
    shm = shared_memory.SharedMemory(create=True, size=Z.nbytes)
    
    # Create a NumPy array backed by the shared memory and copy Z into it
    shared_Z = np.ndarray(Z.shape, dtype=Z.dtype, buffer=shm.buf)
    np.copyto(shared_Z, Z)

    # Determine row indices for each worker chunk
    chunk_size = int(np.ceil(N / num_workers))
    tasks = []
    
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, N)
        
        # Only create a task if there are remaining rows to process
        if start_idx < N:
            tasks.append((shm.name, Z.shape, Z.dtype, start_idx, end_idx, T))

    # Multiprocessing pool
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_compute_chunk, tasks)

    # Combine partial blocks into the full correlation matrix
    C = np.vstack(results)

    # Critically important: Clean up and release the shared memory
    shm.close()
    shm.unlink()

    return C
