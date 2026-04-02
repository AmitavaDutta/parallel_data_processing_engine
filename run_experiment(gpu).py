import os
import argparse
import pandas as pd

# -------------------------------------------------------
# Argument Parsing
# -------------------------------------------------------
parser = argparse.ArgumentParser(description="Run CPU/GPU correlation matrix benchmarks")
parser.add_argument("--mode", type=str, default="cpu", choices=["cpu", "gpu"])
parser.add_argument("--version", type=str, default="baseline", choices=["baseline", "optimized"])
parser.add_argument("--blas", type=str, default="single", choices=["single", "multi"])
parser.add_argument("--dataset", type=str, default="random", choices=["random", "real"])
parser.add_argument("--data_path", type=str, default="src/data/global_temp_use.csv")
parser.add_argument("--data_id", type=str, default=None)
args = parser.parse_args()

# -------------------------------------------------------
# BLAS Thread Control (CPU only)
# -------------------------------------------------------
if args.mode == "cpu" and args.blas == "single":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

# -------------------------------------------------------
# Imports AFTER setting env
# -------------------------------------------------------
from src.cpu.benchmark import run_benchmark
from src.dataset import generate_dataset, read_dataset
from src.cpu.visualize import generate_all_plots
from src.cpu.serial_cpu import compute_correlation_serial

# -------------------------------------------------------
# CPU Experiment
# -------------------------------------------------------
def run_cpu_experiments():
    # Determine dataset name for folder
    dataset_name = args.dataset
    if args.dataset == "real":
        if args.data_id is None:
            raise ValueError("For real dataset, provide --data_id <name>")
        dataset_name = args.data_id

    print(f"\nRunning CPU Experiments | version={args.version} | blas={args.blas} | dataset={dataset_name}\n")

    # Directory structure
    base_dir = f"results/cpu/blas_{args.blas}/{args.version}/{dataset_name}"
    os.makedirs(base_dir, exist_ok=True)

    num_workers = 4
    block_size = 1000
    version = args.version

    # ---------------------
    # Load dataset
    # ---------------------
    if args.dataset == "real":
        X_full = read_dataset(args.data_path)
        print(f"Loaded dataset: shape = {X_full.shape}")
        N_full, T = X_full.shape
        N_values = sorted(list(set([2, 5, 10, N_full])))
    else:
        T = 2000
        N_values = [500,1000,2000,4000,8000,10000,15000,20000]

    results_list = []

    # ---------------------
    # Run benchmarks
    # ---------------------
    for N in N_values:
        X = generate_dataset(N, T) if args.dataset == "random" else X_full[:N, :T]

        result = run_benchmark(N=N, T=T, X=X, num_workers=num_workers,
                               block_size=block_size, version=version)
        result["version"] = version
        result["blas_mode"] = args.blas
        result["dataset"] = dataset_name
        results_list.append(result)

    results_df = pd.DataFrame(results_list)

    # ---------------------
    # Save CSV
    # ---------------------
    results_csv_path = f"{base_dir}/results_{dataset_name}_{version}_{args.blas}.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to {results_csv_path}")

    # ---------------------
    # Example correlation for plots
    # ---------------------
    example_N = min(200, N_values[-1])
    X_example = generate_dataset(example_N, T) if args.dataset == "random" else X_full[:example_N, :T]
    C = compute_correlation_serial(X_example)

    generate_all_plots(results_df, C, output_dir=base_dir, mode="cpu", version=version)

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
    run_cpu_experiments()
elif args.mode == "gpu":
    run_gpu_experiments()
