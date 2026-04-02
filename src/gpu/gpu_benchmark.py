import numpy as np
import torch

from src.gpu.gpu_correlation import (
    gpu_correlation_full,
    gpu_correlation_blockwise
)


def run_gpu_benchmark(N, T, X, block_size, version, device):
    data = X.astype(np.float32)

    if version == "baseline":
        _, full_time, full_mem = gpu_correlation_full(data, device)

        result = {
            "N": N,
            "T": T,
            "gpu_full_time": full_time,
            "gpu_full_mem": full_mem,
            "gpu_block_time": None,
            "gpu_block_mem": None,
            "block_size": None,
        }

    else:
        _, full_time, full_mem = gpu_correlation_full(data, device)
        _, block_time, block_mem = gpu_correlation_blockwise(data, device, block_size)

        result = {
            "N": N,
            "T": T,
            "gpu_full_time": full_time,
            "gpu_full_mem": full_mem,
            "gpu_block_time": block_time,
            "gpu_block_mem": block_mem,
            "block_size": block_size,
        }

    torch.cuda.empty_cache()
    return result
