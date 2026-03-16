"""
run_experiment.py

Unified experiment runner for CPU and GPU correlation matrix benchmarks.

Usage:
python run_experiment.py --mode cpu
python run_experiment.py --mode gpu

CPU Implementations Compared
1. Serial CPU (single-threaded NumPy)
2. Multiprocessing CPU (process-level parallelism)
3. Block-wise CPU (memory-efficient tiled computation)
"""

import os

# Force NumPy BLAS to single thread
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import pandas as pd

from src.cpu.benchmark import run_benchmark
from src.cpu.dataset import generate_dataset
from src.cpu.serial_cpu import compute_correlation_serial
from src.cpu.visualize import generate_all_plots


# -------------------------------------------------------
# CPU Experiment
# -------------------------------------------------------

def run_cpu_experiments():
    """
    Run CPU benchmarks comparing three implementations:
    1. Serial CPU
    2. Multiprocessing CPU
    3. Block-wise CPU
    """

    print("\nRunning CPU Experiments\n")

    os.makedirs("results", exist_ok=True)

    # Experiment parameters
    N_values = [500, 1000, 2000, 4000, 8000, 10000]
    T = 2000
    num_workers = 4
    block_size = 1000

    results = []

    for N in N_values:
        result = run_benchmark(
            N=N,
            T=T,
            num_workers=num_workers,
            block_size=block_size
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    print("\nFinal Benchmark Results\n")
    print(results_df)

    # Save results
    results_df.to_csv("results/results_cpu.csv", index=False)

    # Generate example correlation matrix
    X = generate_dataset(200, T)
    C = compute_correlation_serial(X)

    generate_all_plots(results_df, C)

    print("\nResults saved to results/results_cpu.csv")
    print("Plots saved to /results directory")


# -------------------------------------------------------
# GPU Placeholder
# -------------------------------------------------------

def run_gpu_experiments():
    """
    Placeholder for GPU experiments.
    """

    print("\nGPU mode selected\n")
    print("GPU implementation has not been added yet.")
    print("Expected implementations:")
    print(" - Full GPU correlation")
    print(" - Block-wise GPU correlation")


# -------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run CPU/GPU correlation matrix benchmarks"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Execution mode"
    )

    args = parser.parse_args()

    if args.mode == "cpu":
        run_cpu_experiments()
    elif args.mode == "gpu":
        run_gpu_experiments()

