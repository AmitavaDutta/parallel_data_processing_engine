import os
import argparse
import pandas as pd

# -------------------------------------------------------
# Argument Parsing FIRST (needed before setting BLAS)
# -------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Run CPU/GPU correlation matrix benchmarks"
)

parser.add_argument(
    "--mode",
    type=str,
    default="cpu",
    choices=["cpu", "gpu"],
)

parser.add_argument(
    "--version",
    type=str,
    default="baseline",
    choices=["baseline", "optimized"],
)

parser.add_argument(
    "--blas",
    type=str,
    default="single",
    choices=["single", "multi"],
)

args = parser.parse_args()

# -------------------------------------------------------
# BLAS Thread Control
# -------------------------------------------------------

if args.blas == "single":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

# -------------------------------------------------------
# Imports AFTER setting env
# -------------------------------------------------------

from src.cpu.benchmark import run_benchmark
from src.cpu.dataset import generate_dataset
from src.cpu.serial_cpu import compute_correlation_serial
from src.cpu.visualize import generate_all_plots


# -------------------------------------------------------
# CPU Experiment
# -------------------------------------------------------

def run_cpu_experiments(version, blas_mode):

    print(f"\nRunning CPU Experiments | version={version} | blas={blas_mode}\n")

    # Directory structure
    base_dir = f"results/cpu/blas_{blas_mode}/{version}"
    os.makedirs(base_dir, exist_ok=True)

    # Parameters
    N_values = [500,1000,2000,4000,8000,10000,15000,20000]
    T = 2000
    num_workers = 4
    block_size = 1000

    results = []

    for N in N_values:
        result = run_benchmark(
            N=N,
            T=T,
            num_workers=num_workers,
            block_size=block_size,
            version=version   # <-- pass version
        )
        result["version"] = version
        result["blas_mode"] = blas_mode
        results.append(result)

    results_df = pd.DataFrame(results)

    print("\nFinal Benchmark Results\n")
    print(results_df)

    # Save results
    results_path = f"{base_dir}/results.csv"
    results_df.to_csv(results_path, index=False)

    # Example matrix
    X = generate_dataset(200, T)
    C = compute_correlation_serial(X)

    generate_all_plots(results_df, C, output_dir=base_dir, version=version)


    print(f"\nResults saved to {results_path}")


# -------------------------------------------------------
# GPU Placeholder
# -------------------------------------------------------

def run_gpu_experiments():
    print("\nGPU mode selected\n")
    print("Not implemented")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

if args.mode == "cpu":
    run_cpu_experiments(args.version, args.blas)
elif args.mode == "gpu":
    run_gpu_experiments()

