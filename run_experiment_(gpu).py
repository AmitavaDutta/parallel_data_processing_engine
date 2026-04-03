import os
import types
import pandas as pd
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Config  ← edit these values before running
# -------------------------------------------------------
args = types.SimpleNamespace(
    version    = "baseline",        # "baseline" or "optimized"
    dataset    = "random",          # "random"   or "real"
    data_path  = "src/data/global_temp_use.csv",
    data_id    = None,              # e.g. "global_temp"  (only used when dataset="real")
    block_size = 1024,
)

# -------------------------------------------------------
# Imports
# -------------------------------------------------------
from src.dataset import generate_dataset, read_dataset


# -------------------------------------------------------
# Device Selection
# -------------------------------------------------------
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


# -------------------------------------------------------
# GPU Full-Matrix Correlation
# -------------------------------------------------------
def gpu_correlation_full(data: np.ndarray, device: torch.device):
    """Compute the entire N×N correlation matrix in a single GPU operation."""
    if device.type != "cuda":
        return None, 0, 0

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    X    = torch.tensor(data, device=device)
    mean = X.mean(dim=1, keepdim=True)
    std  = X.std(dim=1, keepdim=True, unbiased=False)
    std  = torch.clamp(std, min=1e-8)
    Z    = (X - mean) / std

    T = data.shape[1]
    corr_matrix = torch.mm(Z, Z.T) / T
    corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    exec_time   = t1 - t0
    peak_mem_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)

    print(f"[GPU Full]  N={data.shape[0]} | Time: {exec_time:.4f}s | VRAM: {peak_mem_mb:.1f} MB")
    return corr_matrix, exec_time, peak_mem_mb


# -------------------------------------------------------
# GPU Blockwise Correlation
# -------------------------------------------------------
def gpu_correlation_blockwise(data: np.ndarray, device: torch.device, block_size: int = 1024):
    """Compute the N×N correlation matrix in blocks to limit peak GPU memory."""
    if device.type != "cuda":
        return None, 0, 0

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    N, T = data.shape
    X    = torch.tensor(data, device=device)
    mean = X.mean(dim=1, keepdim=True)
    std  = X.std(dim=1, keepdim=True, unbiased=False)
    std  = torch.clamp(std, min=1e-8)
    Z    = (X - mean) / std

    corr_matrix = torch.zeros((N, N), device=device, dtype=torch.float32)
    num_blocks  = (N + block_size - 1) // block_size

    for i in range(num_blocks):
        i_start = i * block_size
        i_end   = min(i_start + block_size, N)
        Zi      = Z[i_start:i_end, :]

        for j in range(i, num_blocks):
            j_start    = j * block_size
            j_end      = min(j_start + block_size, N)
            Zj         = Z[j_start:j_end, :]
            block_corr = torch.mm(Zi, Zj.T) / T

            corr_matrix[i_start:i_end, j_start:j_end] = block_corr
            if i != j:
                corr_matrix[j_start:j_end, i_start:i_end] = block_corr.T

    corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    exec_time   = t1 - t0
    peak_mem_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)

    print(f"[GPU Block] N={data.shape[0]} | Time: {exec_time:.4f}s | VRAM: {peak_mem_mb:.1f} MB")
    return corr_matrix, exec_time, peak_mem_mb


# -------------------------------------------------------
# Benchmark Runner
# -------------------------------------------------------
def run_gpu_benchmark(N, T, X, block_size, version, device):
    """Run GPU strategies for a given dataset X and return a result dict."""
    data = X.astype(np.float32)

    if version == "baseline":
        _, full_time, full_mem = gpu_correlation_full(data, device)
        result = {
            "N":              N,
            "T":              T,
            "gpu_full_time":  full_time,
            "gpu_full_mem":   full_mem,
            "gpu_block_time": None,
            "gpu_block_mem":  None,
            "block_size":     None,
        }

    else:  # optimized: run both full and blockwise
        _, full_time,  full_mem  = gpu_correlation_full(data, device)
        _, block_time, block_mem = gpu_correlation_blockwise(data, device, block_size)
        result = {
            "N":              N,
            "T":              T,
            "gpu_full_time":  full_time,
            "gpu_full_mem":   full_mem,
            "gpu_block_time": block_time,
            "gpu_block_mem":  block_mem,
            "block_size":     block_size,
        }

    torch.cuda.empty_cache()
    return result


# -------------------------------------------------------
# Visualization
# -------------------------------------------------------
def generate_gpu_plots(results_df, C_example, output_dir, version):
    """Generate and save performance plots for GPU results."""
    os.makedirs(output_dir, exist_ok=True)

    N_values    = results_df["N"].tolist()
    full_times  = results_df["gpu_full_time"].tolist()
    full_mems   = results_df["gpu_full_mem"].tolist()

    has_block   = results_df["gpu_block_time"].notna().any()
    block_times = results_df["gpu_block_time"].tolist() if has_block else None
    block_mems  = results_df["gpu_block_mem"].tolist()  if has_block else None

    # ── Plot 1: Execution time vs N ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(N_values, full_times, marker="o", label="GPU Full")
    if has_block:
        ax.plot(N_values, block_times, marker="s", linestyle="--", label="GPU Blockwise")
    ax.set_xlabel("N (number of time series)")
    ax.set_ylabel("Execution Time (s)")
    ax.set_title(f"GPU Execution Time vs N  [{version}]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{output_dir}/gpu_time_vs_N_{version}.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[Plot] Saved → {path}")

    # ── Plot 2: Peak VRAM vs N ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(N_values, full_mems, marker="o", color="tab:orange", label="GPU Full")
    if has_block:
        ax.plot(N_values, block_mems, marker="s", linestyle="--",
                color="tab:green", label="GPU Blockwise")
    ax.set_xlabel("N (number of time series)")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.set_title(f"GPU Peak VRAM vs N  [{version}]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{output_dir}/gpu_vram_vs_N_{version}.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[Plot] Saved → {path}")

    # ── Plot 3: Example correlation heatmap ──────────────────────────────────
    if C_example is not None:
        C_np = C_example.cpu().numpy() if isinstance(C_example, torch.Tensor) else C_example
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(C_np, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title("Example Correlation Matrix (GPU)")
        plt.tight_layout()
        path = f"{output_dir}/gpu_corr_heatmap_{version}.png"
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"[Plot] Saved → {path}")

    # ── Plot 4: Side-by-side bar chart for last N ────────────────────────────
    last_row = results_df.iloc[-1]
    labels   = ["GPU Full"]
    times    = [last_row["gpu_full_time"]]
    mems     = [last_row["gpu_full_mem"]]
    if has_block and pd.notna(last_row["gpu_block_time"]):
        labels.append("GPU Blockwise")
        times.append(last_row["gpu_block_time"])
        mems.append(last_row["gpu_block_mem"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'GPU Performance — N={int(last_row["N"])}, T={int(last_row["T"])}', fontsize=13
    )
    bars1 = ax1.bar(labels, times, color=["#1f77b4", "#ff7f0e"])
    ax1.set_ylabel("Execution Time (s)")
    ax1.set_title("Speed Comparison (Lower is Better)")
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + max(times) * 0.02,
                 f"{h:.3f}s", ha="center", va="bottom")

    bars2 = ax2.bar(labels, mems, color=["#1f77b4", "#ff7f0e"])
    ax2.set_ylabel("Peak VRAM (MB)")
    ax2.set_title("Memory Comparison (Lower is Better)")
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + max(mems) * 0.02,
                 f"{h:.1f} MB", ha="center", va="bottom")

    plt.tight_layout()
    path = f"{output_dir}/gpu_bar_comparison_{version}.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[Plot] Saved → {path}")


# -------------------------------------------------------
# GPU Experiment
# -------------------------------------------------------
def run_gpu_experiments():
    dataset_name = args.dataset
    if args.dataset == "real":
        if args.data_id is None:
            raise ValueError("For real dataset, set data_id in the Config block above.")
        dataset_name = args.data_id

    print(f"\nRunning GPU Experiments | version={args.version} | dataset={dataset_name} | block_size={args.block_size}\n")

    device = get_device()
    if device.type != "cuda":
        raise RuntimeError("No CUDA GPU available — enable one in Colab: Runtime → Change runtime type → T4 GPU.")

    base_dir   = f"results/gpu/{args.version}/{dataset_name}"
    os.makedirs(base_dir, exist_ok=True)

    block_size = args.block_size
    version    = args.version

    if args.dataset == "real":
        X_full = read_dataset(args.data_path)
        print(f"Loaded dataset: shape = {X_full.shape}")
        N_full, T = X_full.shape
        N_values  = sorted(list(set([2, 5, 10, N_full])))
    else:
        T        = 2000
        N_values = [500, 1000, 2000, 4000, 8000, 10000, 15000, 20000]

    results_list = []

    for N in N_values:
        X = generate_dataset(N, T) if args.dataset == "random" else X_full[:N, :T]

        result = run_gpu_benchmark(
            N=N, T=T, X=X,
            block_size=block_size,
            version=version,
            device=device,
        )
        result["version"]    = version
        result["dataset"]    = dataset_name
        result["block_size"] = block_size
        results_list.append(result)

    results_df = pd.DataFrame(results_list)

    results_csv_path = f"{base_dir}/results_{dataset_name}_{version}_gpu.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to {results_csv_path}")

    example_N = min(200, N_values[-1])
    X_example = (
        generate_dataset(example_N, T).astype(np.float32)
        if args.dataset == "random"
        else X_full[:example_N, :T].astype(np.float32)
    )
    C_example, _, _ = gpu_correlation_full(X_example, device)

    generate_gpu_plots(results_df, C_example, output_dir=base_dir, version=version)

    torch.cuda.empty_cache()


# -------------------------------------------------------
# Main
# -------------------------------------------------------
run_gpu_experiments()
