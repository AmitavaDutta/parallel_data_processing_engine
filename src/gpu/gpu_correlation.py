import torch
import numpy as np
import time


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Device] GPU detected: {gpu_name} ({total_mem:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[Device] No GPU detected — running on CPU.")
    return device


def gpu_correlation_full(data: np.ndarray, device: torch.device):
    if device.type != "cuda":
        return None, 0, 0

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    X = torch.tensor(data, device=device)
    mean = X.mean(dim=1, keepdim=True)
    std = torch.std(X, dim=1, keepdim=True, unbiased=False)
    std = torch.clamp(std, min=1e-8)
    Z = (X - mean) / std

    T = data.shape[1]
    C = torch.mm(Z, Z.T) / T
    C = torch.clamp(C, -1.0, 1.0)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return C, t1 - t0, torch.cuda.max_memory_allocated(0) / (1024 ** 2)


def gpu_correlation_blockwise(data: np.ndarray, device: torch.device, block_size=1024):
    if device.type != "cuda":
        return None, 0, 0

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    N, T = data.shape
    X = torch.tensor(data, device=device)

    mean = X.mean(dim=1, keepdim=True)
    std = torch.std(X, dim=1, keepdim=True, unbiased=False)
    std = torch.clamp(std, min=1e-8)
    Z = (X - mean) / std

    C = torch.zeros((N, N), device=device)
    num_blocks = (N + block_size - 1) // block_size

    for i in range(num_blocks):
        i_start = i * block_size
        i_end = min(i_start + block_size, N)
        Zi = Z[i_start:i_end]

        for j in range(i, num_blocks):
            j_start = j * block_size
            j_end = min(j_start + block_size, N)
            Zj = Z[j_start:j_end]

            block = torch.mm(Zi, Zj.T) / T
            C[i_start:i_end, j_start:j_end] = block

            if i != j:
                C[j_start:j_end, i_start:i_end] = block.T

    C = torch.clamp(C, -1.0, 1.0)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return C, t1 - t0, torch.cuda.max_memory_allocated(0) / (1024 ** 2)
