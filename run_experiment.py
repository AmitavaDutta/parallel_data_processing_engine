"""
run_experiment.py

Unified experiment runner for CPU and GPU correlation matrix benchmarks.

Usage:
    python run_experiment.py --mode cpu   # run CPU experiments (default)
    python run_experiment.py --mode gpu   # run GPU experiments (placeholder)
    
Features:
- CPU experiments: serial and parallel implementations
- Benchmarking runtime and memory usage
- Validating numerical correctness
- Saving benchmark results to CSV
- Generating plots: runtime, speedup, memory, correlation matrix
- GPU experiments placeholder for future implementation
"""

import argparse
import os
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
    Run CPU benchmarks for serial and parallel correlation matrix computation.

    Steps:
    1. Iterate over multiple dataset sizes (N_values) with fixed time-series length T.
    2. Compute serial and parallel correlation matrices, measuring runtime and memory.
    3. Validate that parallel computation matches serial results.
    4. Save benchmark results to CSV.
    5. Generate visualizations for runtime, speedup, memory, and a sample correlation matrix.
    """
    print("\nRunning CPU Experiments\n")

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Experiment parameters
    N_values = [500, 1000, 2000, 4000, 8000, 10000]  # Number of parallel time series
    T = 2000                             # Length of each time series
    num_workers = 4                      # Threads for parallel CPU computation

    results = []

    # Loop over dataset sizes
    for N in N_values:
        result = run_benchmark(N, T, num_workers)
        results.append(result)

    results_df = pd.DataFrame(results)

    print("\nFinal Benchmark Results\n")
    print(results_df)

    # Save results to CSV
    results_df.to_csv("results/results_cpu.csv", index=False)

    # Generate a sample correlation matrix for visualization
    X = generate_dataset(200, T)
    C = compute_correlation_serial(X)

    # Generate plots and save to /results
    generate_all_plots(results_df, C)

    print("\nResults saved to results/results_cpu.csv")
    print("Plots saved to /results directory")


# -------------------------------------------------------
# GPU Placeholder
# -------------------------------------------------------
def run_gpu_experiments():
    """
    Placeholder for GPU experiments.

    Expected GPU pipeline (to be implemented):
    1. Dataset generation (same interface as CPU)
    2. GPU correlation computation
    3. Runtime and memory measurement
    4. Validation against CPU results
    5. Benchmarking and visualization
    """
    print("\nGPU mode selected\n")
    print("GPU implementation has not been added yet.")
    print("When implemented, it should follow the same interface as CPU benchmark.")


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
        help="Execution mode: cpu or gpu"
    )

    args = parser.parse_args()

    if args.mode == "cpu":
        run_cpu_experiments()
    elif args.mode == "gpu":
        run_gpu_experiments()

