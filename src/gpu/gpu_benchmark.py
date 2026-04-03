import numpy as np
import torch
import tracemalloc
from src.gpu.gpu_correlation import (
    gpu_correlation_full,
    gpu_correlation_blockwise
)

def clear_gpu_cache():
    """Ensure absolute independence between runs by wiping GPU cache and memory stats."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def run_gpu_benchmark(N, T, X, block_size, version, device):
    data = X.astype(np.float32)

    # Initialize result dictionary for this N, T combination
    result = {
        "N": N,
        "T": T,
        "block_size": block_size if version != "baseline" else None,
    }

    modes = ["pageable", "pinned"]

    for mode in modes:
        # ==========================================
        # 1. RUN FULL STRATEGY (Independently)
        # ==========================================
        clear_gpu_cache() # <-- Wipe GPU memory before starting
        _, h2d_full, comp_full, d2h_full, mem_full = gpu_correlation_full(data, device, mode=mode)
        
        result[f"gpu_full_{mode}_h2d"] = h2d_full
        result[f"gpu_full_{mode}_comp"] = comp_full
        result[f"gpu_full_{mode}_d2h"] = d2h_full
        result[f"gpu_full_{mode}_total"] = (h2d_full + comp_full + d2h_full) if h2d_full is not None else None
        result[f"gpu_full_{mode}_mem"] = mem_full

        # ==========================================
        # 2. RUN BLOCKWISE STRATEGY (Independently)
        # ==========================================
        if version != "baseline":
            clear_gpu_cache() # <-- Wipe GPU memory again
            _, h2d_block, comp_block, d2h_block, mem_block = gpu_correlation_blockwise(
                data, device, block_size=block_size, mode=mode
            )
            
            result[f"gpu_block_{mode}_h2d"] = h2d_block
            result[f"gpu_block_{mode}_comp"] = comp_block
            result[f"gpu_block_{mode}_d2h"] = d2h_block
            result[f"gpu_block_{mode}_total"] = (h2d_block + comp_block + d2h_block) if h2d_block is not None else None
            result[f"gpu_block_{mode}_mem"] = mem_block
        else:
            result[f"gpu_block_{mode}_h2d"] = None
            result[f"gpu_block_{mode}_comp"] = None
            result[f"gpu_block_{mode}_d2h"] = None
            result[f"gpu_block_{mode}_total"] = None
            result[f"gpu_block_{mode}_mem"] = None

    return result


'''
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
'''