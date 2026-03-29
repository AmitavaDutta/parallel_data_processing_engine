import numpy as np
import time
import torch

def gpu_correlation_full(data: np.ndarray, device: torch.device):
    """Compute the entire N×N correlation matrix in a single GPU operation."""
    if device.type != "cuda": return None, 0, 0
    print("\n--- Running GPU Full Matrix ---")

    torch.cuda.reset_peak_memory_stats() # Reset memory tracker
    t0 = time.perf_counter()

    X = torch.tensor(data, device=device)
    mean = X.mean(dim=1, keepdim=True)
    std  = X.std(dim=1, keepdim=True, unbiased=False)
    std  = torch.clamp(std, min=1e-8)
    Z    = (X - mean) / std

    T = data.shape[1]
    # Matrix multiplication for correlation
    corr_matrix = torch.mm(Z, Z.T) / T
    corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)

    torch.cuda.synchronize() # Wait for GPU to finish
    t1 = time.perf_counter()

    exec_time = t1 - t0
    peak_mem_mb = torch.cuda.max_memory_allocated(0) / (1024**2)

    print(f"[GPU Full] Execution Time: {exec_time:.4f}s")
    print(f"[GPU Full] Peak VRAM: {peak_mem_mb:.1f} MB")

    return corr_matrix, exec_time, peak_mem_mb

import numpy as np
import time
import torch

from cancer_data import load_data


def cpu_corr(data):
    start = time.time()
    np.corrcoef(data)
    return time.time() - start


def run_experiment():
    data = load_data()

    # make data bigger
    data = np.tile(data, (10, 1))

    print("Data shape:", data.shape)

    # CPU timing
    cpu_time = cpu_corr(data)

    # GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GPU timing
    _, gpu_time, gpu_mem = gpu_correlation_full(data, device)

    print("\n--- Results ---")
    print(f"CPU time: {cpu_time:.4f}s")

    if gpu_time != 0:
        print(f"GPU time: {gpu_time:.4f}s")
        print(f"GPU memory: {gpu_mem:.2f} MB")
    else:
        print("GPU not available")
