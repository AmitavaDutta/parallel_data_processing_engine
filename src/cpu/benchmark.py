import time
import numpy as np
from memory_profiler import memory_usage

# Use relative imports
from .dataset import generate_dataset
from .serial_cpu import compute_correlation_serial
from .parallel_cpu import parallel_cpu_correlation
#from .visualize import generate_all_plots ## not needed



def measure_memory(func, *args):
    """
    Measure peak memory usage of a function execution.

    Parameters
    ----------
    func : callable
        The function whose memory usage is to be measured.
    *args : any
        Arguments to pass to the function.

    Returns
    -------
    mem_peak : float
        Peak memory usage in MB during the function execution.
    result : any
        Return value of the function.
    """
    # memory_usage returns a tuple (mem_usage, retval) if retval=True
    mem_peak, result = memory_usage((func, args), retval=True, max_usage=True)
    return mem_peak, result


def run_benchmark(N, T, num_workers=4):
    """
    Run a benchmark comparing serial and parallel CPU correlation computations.

    Parameters
    ----------
    N : int
        Number of parallel time series.
    T : int
        Number of time steps per series.
    num_workers : int, optional (default=4)
        Number of threads to use for parallel computation.

    Returns
    -------
    dict
        A dictionary containing benchmark results:
        - N, T
        - serial_time, parallel_time
        - speedup
        - serial_memory_MB, parallel_memory_MB
        - correct (boolean indicating numerical consistency)
    """
    print(f"\nRunning Benchmark: N={N}, T={T}")

    # Generate synthetic dataset
    X = generate_dataset(N, T)

    # ----------------------
    # Serial CPU benchmark
    # ----------------------
    start = time.time()
    mem_serial, C1 = measure_memory(compute_correlation_serial, X)
    serial_time = time.time() - start

    # ----------------------
    # Parallel CPU benchmark
    # ----------------------
    start = time.time()
    mem_parallel, C2 = measure_memory(parallel_cpu_correlation, X, num_workers)
    parallel_time = time.time() - start

    # ----------------------
    # Validation: ensure numerical correctness
    # ----------------------
    correct = np.allclose(C1, C2)

    # Compute speedup
    speedup = serial_time / parallel_time if parallel_time > 0 else 0

    # Aggregate results in a dictionary
    results = {
        "N": N,
        "T": T,
        "serial_time": serial_time,
        "parallel_time": parallel_time,
        "speedup": speedup,
        "serial_memory_MB": mem_serial,
        "parallel_memory_MB": mem_parallel,
        "correct": correct
    }

    return results


if __name__ == "__main__":
    # Quick test benchmark for N=500, T=2000
    results = run_benchmark(500, 2000)

    print("\nBenchmark Results")
    print("-" * 30)
    for key, value in results.items():
        print(f"{key}: {value}")

