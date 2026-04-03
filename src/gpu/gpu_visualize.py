import os
import matplotlib.pyplot as plt
import torch

def plot_runtime(results_df, save_path, version):
    """Line plot comparing the total execution time of all 4 modes."""
    N = results_df["N"]

    plt.figure(figsize=(10, 6))
    
    # Plot Full Strategies
    if "gpu_full_pageable_total" in results_df.columns and results_df["gpu_full_pageable_total"].notna().any():
        plt.plot(N, results_df["gpu_full_pageable_total"], marker="o", label="Full (Pageable)")
    if "gpu_full_pinned_total" in results_df.columns and results_df["gpu_full_pinned_total"].notna().any():
        plt.plot(N, results_df["gpu_full_pinned_total"], marker="o", linestyle="-.", label="Full (Pinned)")

    # Plot Blockwise Strategies
    if "gpu_block_pageable_total" in results_df.columns and results_df["gpu_block_pageable_total"].notna().any():
        plt.plot(N, results_df["gpu_block_pageable_total"], marker="s", linestyle="--", label="Blockwise (Pageable)")
    if "gpu_block_pinned_total" in results_df.columns and results_df["gpu_block_pinned_total"].notna().any():
        plt.plot(N, results_df["gpu_block_pinned_total"], marker="s", linestyle=":", label="Blockwise (Pinned)")

    plt.xlabel("Number of Time Series (N)")
    plt.ylabel("Total Execution Time (s)")
    plt.title(f"GPU Runtime Comparison [{version}]")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_memory(results_df, save_path, version):
    """Line plot comparing peak VRAM consumption."""
    N = results_df["N"]

    plt.figure(figsize=(10, 6))
    
    # Plot Full Strategies Memory
    if "gpu_full_pageable_mem" in results_df.columns and results_df["gpu_full_pageable_mem"].notna().any():
        plt.plot(N, results_df["gpu_full_pageable_mem"], marker="o", label="Full (Pageable)")
    if "gpu_full_pinned_mem" in results_df.columns and results_df["gpu_full_pinned_mem"].notna().any():
        plt.plot(N, results_df["gpu_full_pinned_mem"], marker="o", linestyle="-.", label="Full (Pinned)")

    # Plot Blockwise Strategies Memory
    if "gpu_block_pageable_mem" in results_df.columns and results_df["gpu_block_pageable_mem"].notna().any():
        plt.plot(N, results_df["gpu_block_pageable_mem"], marker="s", linestyle="--", label="Blockwise (Pageable)")
    if "gpu_block_pinned_mem" in results_df.columns and results_df["gpu_block_pinned_mem"].notna().any():
        plt.plot(N, results_df["gpu_block_pinned_mem"], marker="s", linestyle=":", label="Blockwise (Pinned)")

    plt.xlabel("Number of Time Series (N)")
    plt.ylabel("Peak VRAM Usage (MB)")
    plt.title(f"GPU Memory Consumption [{version}]")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_pipeline_breakdown(results_df, save_path, version):
    """Creates a stacked bar chart for the largest N to show H2D vs Compute vs D2H."""
    # Find the row with the largest N that successfully ran full GPU benchmarks
    valid_df = results_df.dropna(subset=["gpu_full_pageable_total"])
    if valid_df.empty:
        return
    
    max_row = valid_df.iloc[-1]
    max_N = int(max_row["N"])

    labels = ['Full\n(Pageable)', 'Full\n(Pinned)', 'Blockwise\n(Pageable)', 'Blockwise\n(Pinned)']
    
    # Extract components
    h2d = [
        max_row.get("gpu_full_pageable_h2d", 0), max_row.get("gpu_full_pinned_h2d", 0),
        max_row.get("gpu_block_pageable_h2d", 0), max_row.get("gpu_block_pinned_h2d", 0)
    ]
    comp = [
        max_row.get("gpu_full_pageable_comp", 0), max_row.get("gpu_full_pinned_comp", 0),
        max_row.get("gpu_block_pageable_comp", 0), max_row.get("gpu_block_pinned_comp", 0)
    ]
    d2h = [
        max_row.get("gpu_full_pageable_d2h", 0), max_row.get("gpu_full_pinned_d2h", 0),
        max_row.get("gpu_block_pageable_d2h", 0), max_row.get("gpu_block_pinned_d2h", 0)
    ]

    # Clean up None values (in case a run crashed/skipped)
    h2d = [x if x is not None else 0 for x in h2d]
    comp = [x if x is not None else 0 for x in comp]
    d2h = [x if x is not None else 0 for x in d2h]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Stack the bars
    ax.bar(labels, h2d, label='H2D (Host to Device)', color='#1f77b4')
    ax.bar(labels, comp, bottom=h2d, label='Compute (Pure Math)', color='#ff7f0e')
    bottom_d2h = [i+j for i,j in zip(h2d, comp)]
    ax.bar(labels, d2h, bottom=bottom_d2h, label='D2H (Device to Host)', color='#2ca02c')

    ax.set_ylabel('Time (Seconds)')
    ax.set_title(f'Pipeline Overhead Breakdown for N={max_N} [{version}]')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_correlation_matrix(C, save_path):
    """Visualizes the final correlation matrix."""
    if C is None:
        return

    C_np = C.cpu().numpy() if isinstance(C, torch.Tensor) else C

    plt.figure()
    plt.imshow(C_np, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("GPU Correlation Matrix")
    plt.savefig(save_path)
    plt.close()


def generate_all_plots(results_df, C, output_dir="results", mode="gpu", version="baseline", file_prefix=""):
    os.makedirs(output_dir, exist_ok=True)
    
    # Standard plots
    plot_runtime(results_df, f"{output_dir}/{file_prefix}_{mode}_runtime.png", version)
    plot_memory(results_df, f"{output_dir}/{file_prefix}_{mode}_memory.png", version)
    plot_correlation_matrix(C, f"{output_dir}/{file_prefix}_{mode}_correlation_matrix.png")
    
    # NEW: Pipeline overhead breakdown
    plot_pipeline_breakdown(results_df, f"{output_dir}/{file_prefix}_{mode}_pipeline_breakdown.png", version)

'''
def plot_runtime(results_df, save_path, version):
    N = results_df["N"]

    plt.figure()
    plt.plot(N, results_df["gpu_full_time"], marker="o", label="GPU Full")

    if results_df["gpu_block_time"].notna().any():
        plt.plot(N, results_df["gpu_block_time"], marker="s", linestyle="--", label="GPU Block")

    plt.xlabel("N")
    plt.ylabel("Execution Time (s)")
    plt.title(f"GPU Runtime [{version}]")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()


def plot_memory(results_df, save_path, version):
    N = results_df["N"]

    plt.figure()
    plt.plot(N, results_df["gpu_full_mem"], marker="o", label="GPU Full")

    if results_df["gpu_block_mem"].notna().any():
        plt.plot(N, results_df["gpu_block_mem"], marker="s", linestyle="--", label="GPU Block")

    plt.xlabel("N")
    plt.ylabel("VRAM (MB)")
    plt.title(f"GPU Memory [{version}]")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()


def plot_correlation_matrix(C, save_path):
    if C is None:
        return

    C_np = C.cpu().numpy() if isinstance(C, torch.Tensor) else C

    plt.figure()
    plt.imshow(C_np, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("GPU Correlation Matrix")
    plt.savefig(save_path)
    plt.close()


def generate_all_plots(results_df, C, output_dir="results", mode="gpu", version="baseline", file_prefix=""):
    os.makedirs(output_dir, exist_ok=True)
    plot_runtime(results_df, f"{output_dir}/{file_prefix}_{mode}_runtime.png", version)
    plot_memory(results_df, f"{output_dir}/{file_prefix}_{mode}_memory.png", version)
    plot_correlation_matrix(C, f"{output_dir}/{file_prefix}_{mode}_correlation_matrix.png")
'''