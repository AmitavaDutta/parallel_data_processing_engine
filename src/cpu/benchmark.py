import time
import numpy as np
import psutil
import os

# Relative imports
from .dataset import generate_dataset
from .serial_cpu import compute_correlation_serial
from .parallel_cpu import parallel_cpu_correlation
from .block_cpu import compute_correlation_blockwise


def measure_memory(func, *args):
    """
    Measure memory usage of a function execution using psutil.

    Parameters
    ----------
    func : callable
        Function to execute.
    *args :
        Arguments to pass to the function.

    Returns
    -------
    mem_usage_MB : float
        Approximate memory usage of the process during execution (MB).
    result : any
        Return value of the function.
    """

    process = psutil.Process(os.getpid())

    mem_before = process.memory_info().rss / (1024 ** 2)

    result = func(*args)

    mem_after = process.memory_info().rss / (1024 ** 2)

    mem_usage_MB = max(mem_before, mem_after)

    return mem_usage_MB, result


def run_benchmark(N, T, num_workers=4, block_size=1000):
    """
    Run benchmark comparing serial, parallel, and block-wise CPU implementations.

    Parameters
    ----------
    N : int
        Number of parallel time series.
    T : int
        Number of time steps per series.
    num_workers : int
        Number of processes for multiprocessing.
    block_size : int
        Block size for block-wise computation.

    Returns
    -------
    dict
        Benchmark results.
    """

    print(f"\nRunning Benchmark: N={N}, T={T}")

    # Generate dataset
    X = generate_dataset(N, T)

    # ---------------------
    # Serial CPU benchmark
    # ---------------------
    start = time.time()
    mem_serial, C_serial = measure_memory(compute_correlation_serial, X)
    serial_time = time.time() - start

    # ---------------------
    # Parallel CPU benchmark
    # ---------------------
    start = time.time()
    mem_parallel, C_parallel = measure_memory(
        parallel_cpu_correlation, X, num_workers
    )
    parallel_time = time.time() - start

    # ---------------------
    # Block-wise CPU benchmark
    # ---------------------
    start = time.time()
    mem_block, C_block = measure_memory(
        compute_correlation_blockwise, X, block_size
    )
    block_time = time.time() - start

    # ---------------------
    # Validation
    # ---------------------
    correct_parallel = np.allclose(C_serial, C_parallel)
    correct_block = np.allclose(C_serial, C_block)

    # Speedups
    parallel_speedup = serial_time / parallel_time if parallel_time > 0 else 0
    block_speedup = serial_time / block_time if block_time > 0 else 0

    results = {
        "N": N,
        "T": T,

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


if __name__ == "__main__":

    results = run_benchmark(500, 2000)

    print("\nBenchmark Results")
    print("-" * 40)

    for key, value in results.items():
        print(f"{key}: {value}")

