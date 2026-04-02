"""
GPU vs CPU Correlation Matrix Computation & Benchmarking
DS Practice Project

This script calculates an N×N correlation matrix for N time series.
It has the following two approaches:
1. GPU (PyTorch Full) - Computes the entire matrix in one operation.
2. GPU (PyTorch Blockwise) - Computes the matrix in chunks to save VRAM.

It then visualizes the time and memory differences using Matplotlib.
"""

import torch
import numpy as np
import time
import tracemalloc  # For tracking CPU memory usage
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Device Selection
# ─────────────────────────────────────────────────────────────────────────────
# This block checks if your machine has an NVIDIA GPU configured with CUDA.
# PyTorch needs to know whether to send operations to the 'cuda' device or
# fall back to the standard 'cpu'.
def get_device():
    """Detect whether a CUDA-compatible GPU is available."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Device] GPU detected: {gpu_name} ({total_mem:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[Device] No GPU detected — running on CPU.")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Synthetic Dataset Generation
# ─────────────────────────────────────────────────────────────────────────────
# To benchmark the hardware, we need raw data. This function uses NumPy to
# generate N random time series, each with T time steps. We use float32
# because GPUs process 32-bit floats much faster than 64-bit floats.
def generate_time_series(N: int, T: int, seed: int = 42) -> np.ndarray:
    """Generate a random dataset of N time series, each of length T."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((N, T)).astype(np.float32)
    print(f"[Data]   Generated {N} time series of length {T} "
          f"({data.nbytes / 1024**2:.1f} MB as float32)")
    return data

'''
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: CPU Baseline Correlation
# ─────────────────────────────────────────────────────────────────────────────
# This acts as our baseline. We use NumPy's built-in `corrcoef` function
# to calculate the correlation matrix entirely on the CPU. We also use
# `tracemalloc` to monitor how much RAM the CPU uses during this calculation.
def cpu_correlation(data: np.ndarray):
    """Compute correlation matrix on CPU using NumPy and track memory/time."""
    print("\n--- Running CPU Baseline ---")
    tracemalloc.start()  # Start tracking CPU memory
    t0 = time.perf_counter()

    # Calculate correlation matrix using NumPy
    corr_matrix = np.corrcoef(data)

    t1 = time.perf_counter()

    # Get peak memory usage during the operation
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    exec_time = t1 - t0
    peak_mem_mb = peak_mem_bytes / (1024 * 1024)

    print(f"[CPU] Execution Time: {exec_time:.4f}s")
    print(f"[CPU] Peak Memory: {peak_mem_mb:.1f} MB")

    return corr_matrix, exec_time, peak_mem_mb

'''
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Full-Matrix GPU Correlation (Strategy A)
# ─────────────────────────────────────────────────────────────────────────────
# This strategy moves the data to the GPU and computes the entire N×N matrix
# at once using PyTorch. It standardizes the data (Z-score) and then performs
# a matrix multiplication (Z @ Z.T). This is the fastest method but uses the
# most VRAM, which can crash if N is extremely large.
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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Block-Wise GPU Correlation (Strategy B)
# ─────────────────────────────────────────────────────────────────────────────
# If N is huge (e.g., 20,000 series), the resulting matrix won't fit in VRAM.
# This function calculates the matrix in small "blocks". It iterates over chunks
# of the data, computes a small piece of the final matrix, and slots it into place.
# It trades a tiny bit of speed for massive memory savings.
def gpu_correlation_blockwise(data: np.ndarray, device: torch.device, block_size: int = 512):
    """Compute the N×N correlation matrix in blocks to limit peak GPU memory."""
    if device.type != "cuda": return None, 0, 0
    print(f"\n--- Running GPU Blockwise (Block Size: {block_size}) ---")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    N, T = data.shape
    X    = torch.tensor(data, device=device)
    mean = X.mean(dim=1, keepdim=True)
    std  = X.std(dim=1, keepdim=True, unbiased=False)
    std  = torch.clamp(std, min=1e-8)
    Z    = (X - mean) / std

    # Pre-allocate the final matrix
    corr_matrix = torch.zeros((N, N), device=device, dtype=torch.float32)
    num_blocks = (N + block_size - 1) // block_size

    for i in range(num_blocks):
        i_start = i * block_size
        i_end   = min(i_start + block_size, N)
        Zi      = Z[i_start:i_end, :]

        for j in range(i, num_blocks):
            j_start = j * block_size
            j_end   = min(j_start + block_size, N)
            Zj      = Z[j_start:j_end, :]

            block_corr = torch.mm(Zi, Zj.T) / T

            # Write to upper and lower triangles
            corr_matrix[i_start:i_end, j_start:j_end] = block_corr
            if i != j:
                corr_matrix[j_start:j_end, i_start:i_end] = block_corr.T

    corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    exec_time = t1 - t0
    peak_mem_mb = torch.cuda.max_memory_allocated(0) / (1024**2)

    print(f"[GPU Block] Execution Time: {exec_time:.4f}s")
    print(f"[GPU Block] Peak VRAM: {peak_mem_mb:.1f} MB")

    return corr_matrix, exec_time, peak_mem_mb


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Visualization
# ─────────────────────────────────────────────────────────────────────────────
# This function takes the timing and memory data collected from the previous
# blocks and creates a Matplotlib figure with two bar charts side-by-side.
# It makes it visually obvious how much faster (or memory-efficient) the GPU is.
def plot_comparisons(N, T, times, memories):
    """Generate bar charts to help visualise GPU performance."""
    labels = ['GPU (Full)', 'GPU (Blockwise)']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Performance Comparison: {N} Time Series, {T} Steps', fontsize=14)

    # Plot 1: Execution Time
    bars1 = ax1.bar(labels, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Speed Comparison (Lower is Better)')
    # Add text labels on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + (max(times)*0.02), f'{yval:.3f}s', ha='center', va='bottom')

    # Plot 2: Peak Memory
    bars2 = ax2.bar(labels, memories, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_ylabel('Peak Memory Usage (MB)')
    ax2.set_title('Memory Comparison (Lower is Better)')
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + (max(memories)*0.02), f'{yval:.1f} MB', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Main Experiment Runner
# ─────────────────────────────────────────────────────────────────────────────
# The central controller. It generates the data, calls the CPU and GPU
# functions, stores the results, and then triggers the plotting function.
def run_experiment(N: int, T: int, block_size: int = 512):
    print("\n" + "═" * 60)
    print(f"  Experiment: N={N} time series, T={T} time steps")
    print("═" * 60)

    device = get_device()
    data = generate_time_series(N, T)

    # Run CPU
    #_, cpu_time, cpu_mem = cpu_correlation(data)

    if device.type == "cuda":
        # Run GPU Full
        _, gpu_f_time, gpu_f_mem = gpu_correlation_full(data, device)

        # Run GPU Blockwise
        _, gpu_b_time, gpu_b_mem = gpu_correlation_blockwise(data, device, block_size)

        # Plot Results
        times = [gpu_f_time, gpu_b_time]
        memories = [gpu_f_mem, gpu_b_mem]
        plot_comparisons(N, T, times, memories)

        # Cleanup
        torch.cuda.empty_cache()
    else:
        print("\n[Notice] Visualizations skipped because no GPU is available to compare against.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
# This is where the script begins execution.
if __name__ == "__main__":
    # You can change these numbers. N=5000 might take a few seconds on CPU.
    run_experiment(N=5000, T=1000, block_size=1024)
