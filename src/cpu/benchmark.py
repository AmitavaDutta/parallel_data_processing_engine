import time
import numpy as np
from memory_profiler import memory_usage

# Dataset
from src.dataset import generate_dataset

# Serial
from .serial_cpu import compute_correlation_serial

# Parallel (baseline + optimized)
from .parallel_cpu import (
    parallel_cpu_correlation_baseline,
    parallel_cpu_correlation_optimized
)

# Block (baseline + optimized)
from .block_cpu import (
    compute_correlation_blockwise_baseline,
    compute_correlation_blockwise_optimized
)

# Visualization
from .visualize import generate_all_plots


# -------------------------------------------------------
# Peak Memory Measurement
# -------------------------------------------------------
def measure_memory(func, *args):
    """
    Measure peak memory usage during execution.
    Returns peak memory (MB) and function result.
    """
    result_container = {}

    def wrapper():
        result_container["result"] = func(*args)

    peak_memory_MB = memory_usage(
        (wrapper,),
        interval=0.01,
        retval=False,
        max_usage=True
    )

    return peak_memory_MB, result_container["result"]


# -------------------------------------------------------
# Benchmark Runner
# -------------------------------------------------------
def run_benchmark(N=None, T=None, X=None, num_workers=4, block_size=1000, version="baseline"):
    """
    Run benchmark comparing serial, parallel, and block-wise CPU implementations.
    Supports baseline and optimized versions.

    Parameters
    ----------
    N, T : int
        Number of time series and time steps (used only if X is None)
    X : np.ndarray
        Optional precomputed dataset (real or subset of real)
    num_workers : int
    block_size : int
    version : str
        'baseline' or 'optimized'

    Returns
    -------
    dict
        Benchmark results including times, memory, speedups, and correctness.
    """

    # ---------------------
    # Dataset
    # ---------------------
    if X is None:
        if N is None or T is None:
            raise ValueError("Either provide N and T for random data or X for precomputed data")
        X = generate_dataset(N, T)
    else:
        N, T = X.shape

    print(f"\nRunning Benchmark: N={N}, T={T}, version={version}")

    # ---------------------
    # Select Implementation
    # ---------------------
    if version == "baseline":
        parallel_func = parallel_cpu_correlation_baseline
        block_func = compute_correlation_blockwise_baseline
    else:
        parallel_func = parallel_cpu_correlation_optimized
        block_func = compute_correlation_blockwise_optimized

    # ---------------------
    # Serial
    # ---------------------
    start = time.time()
    mem_serial, C_serial = measure_memory(compute_correlation_serial, X)
    serial_time = time.time() - start

    # ---------------------
    # Parallel
    # ---------------------
    start = time.time()
    mem_parallel, C_parallel = measure_memory(parallel_func, X, num_workers)
    parallel_time = time.time() - start

    # ---------------------
    # Block-wise
    # ---------------------
    start = time.time()
    mem_block, C_block = measure_memory(block_func, X, block_size)
    block_time = time.time() - start

    # ---------------------
    # Validation (subset to speed up)
    # ---------------------
    subset = min(100, N)
    correct_parallel = np.allclose(C_serial[:subset, :subset], C_parallel[:subset, :subset])
    correct_block = np.allclose(C_serial[:subset, :subset], C_block[:subset, :subset])

    # ---------------------
    # Speedups
    # ---------------------
    parallel_speedup = serial_time / parallel_time if parallel_time > 0 else 0
    block_speedup = serial_time / block_time if block_time > 0 else 0

    # ---------------------
    # Results
    # ---------------------
    results = {
        "N": N,
        "T": T,
        "version": version,

        "serial_time": serial_time,
        "parallel_time": parallel_time,
        "block_time": block_time,

        "parallel_speedup": parallel_speedup,
        "block_speedup": block_speedup,

        "serial_memory_MB": mem_serial,
        "parallel_memory_MB": mem_parallel,
        "block_memory_MB": mem_block,

        "correct_parallel": correct_parallel,
        "correct_block": correct_block
    }

    return results
