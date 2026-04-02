import os
import matplotlib.pyplot as plt
import torch


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
