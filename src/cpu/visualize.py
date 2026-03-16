"""
visualize.py

Visualization utilities for CPU correlation matrix benchmark results.

Generates plots for:
1. Runtime vs number of time series
2. Speedup vs number of time series
3. Memory usage vs number of time series
4. Example correlation matrix heatmap
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os


# -------------------------------------------------------
# Runtime Plot
# -------------------------------------------------------

def plot_runtime(results_df, save_path):

    plt.figure(figsize=(8,6))

    plt.plot(
        results_df["N"],
        results_df["serial_time"],
        marker="o",
        label="Serial CPU"
    )

    plt.plot(
        results_df["N"],
        results_df["parallel_time"],
        marker="s",
        label="Multiprocessing CPU"
    )

    plt.plot(
        results_df["N"],
        results_df["block_time"],
        marker="^",
        label="Block-wise CPU"
    )

    plt.xlabel("Number of Time Series (N)")
    plt.ylabel("Runtime (seconds)")
    plt.title("CPU Runtime Comparison")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()


# -------------------------------------------------------
# Speedup Plot
# -------------------------------------------------------

def plot_speedup(results_df, save_path):

    plt.figure(figsize=(8,6))

    plt.plot(
        results_df["N"],
        results_df["parallel_speedup"],
        marker="o",
        label="Multiprocessing CPU"
    )

    plt.plot(
        results_df["N"],
        results_df["block_speedup"],
        marker="s",
        label="Block-wise CPU"
    )

    plt.xlabel("Number of Time Series (N)")
    plt.ylabel("Speedup vs Serial CPU")
    plt.title("CPU Parallel Speedup")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()


# -------------------------------------------------------
# Memory Plot
# -------------------------------------------------------

def plot_memory(results_df, save_path):

    plt.figure(figsize=(8,6))

    plt.plot(
        results_df["N"],
        results_df["serial_memory_MB"],
        marker="o",
        label="Serial CPU"
    )

    plt.plot(
        results_df["N"],
        results_df["parallel_memory_MB"],
        marker="s",
        label="Multiprocessing CPU"
    )

    plt.plot(
        results_df["N"],
        results_df["block_memory_MB"],
        marker="^",
        label="Block-wise CPU"
    )

    plt.xlabel("Number of Time Series (N)")
    plt.ylabel("Memory Usage (MB)")
    plt.title("CPU Memory Usage Comparison")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()


# -------------------------------------------------------
# Correlation Matrix Heatmap
# -------------------------------------------------------

def plot_correlation_matrix(C, save_path):

    plt.figure(figsize=(8,6))

    sns.heatmap(
        C,
        cmap="coolwarm",
        center=0,
        square=True,
        cbar=True
    )

    plt.title("Sample Correlation Matrix")

    plt.savefig(save_path)
    plt.close()


# -------------------------------------------------------
# Generate All Plots
# -------------------------------------------------------

def generate_all_plots(results_df, C, output_dir="results", mode="cpu"):

    os.makedirs(output_dir, exist_ok=True)

    plot_runtime(
        results_df,
        f"{output_dir}/{mode}_runtime.png"
    )

    plot_speedup(
        results_df,
        f"{output_dir}/{mode}_speedup.png"
    )

    plot_memory(
        results_df,
        f"{output_dir}/{mode}_memory.png"
    )

    plot_correlation_matrix(
        C,
        f"{output_dir}/{mode}_correlation_matrix.png"
    )

