"""
visualize.py

Functions for visualizing correlation matrix computation results, including:
- Correlation heatmaps
- Runtime scaling plots
- Speedup plots
- Memory usage plots

Can also generate all plots at once from a results DataFrame.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


# -------------------------------------------------------
# Correlation Matrix Visualization
# -------------------------------------------------------
def plot_correlation_matrix(C, save_path=None):
    """
    Plot a heatmap of the correlation matrix.

    Parameters
    ----------
    C : np.ndarray
        The correlation matrix to visualize (N x N).
    save_path : str or None
        Path to save the figure. If None, the figure is not saved.
    """
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        C,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Correlation"}
    )

    plt.title("Correlation Matrix")
    plt.xlabel("Time Series Index")
    plt.ylabel("Time Series Index")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


# -------------------------------------------------------
# Runtime Comparison Plot
# -------------------------------------------------------
def plot_runtime(results_df, save_path=None):
    """
    Plot runtime of serial vs parallel CPU implementations.

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results containing columns: "N", "serial_time", "parallel_time".
    save_path : str or None
        Path to save the figure. If None, the figure is not saved.
    """
    plt.figure(figsize=(8, 5))

    plt.plot(
        results_df["N"],
        results_df["serial_time"],
        marker="o",
        linewidth=2,
        label="Serial CPU"
    )
    plt.plot(
        results_df["N"],
        results_df["parallel_time"],
        marker="o",
        linewidth=2,
        label="Parallel CPU"
    )

    plt.xlabel("Number of Time Series (N)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Scaling")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


# -------------------------------------------------------
# Speedup Plot
# -------------------------------------------------------
def plot_speedup(results_df, save_path=None):
    """
    Plot speedup of parallel CPU relative to serial CPU.

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results containing columns: "N", "speedup".
    save_path : str or None
        Path to save the figure. If None, the figure is not saved.
    """
    plt.figure(figsize=(8, 5))

    plt.plot(
        results_df["N"],
        results_df["speedup"],
        marker="o",
        linewidth=2
    )
    plt.axhline(1, linestyle="--", color="gray")  # baseline for no speedup

    plt.xlabel("Number of Time Series (N)")
    plt.ylabel("Speedup (Serial / Parallel)")
    plt.title("Parallel Speedup")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


# -------------------------------------------------------
# Memory Usage Plot
# -------------------------------------------------------
def plot_memory(results_df, save_path=None):
    """
    Plot memory usage of serial vs parallel CPU implementations.

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results containing columns: "N", "serial_memory_MB", "parallel_memory_MB".
    save_path : str or None
        Path to save the figure. If None, the figure is not saved.
    """
    plt.figure(figsize=(8, 5))

    plt.plot(
        results_df["N"],
        results_df["serial_memory_MB"],
        marker="o",
        linewidth=2,
        label="Serial CPU"
    )
    plt.plot(
        results_df["N"],
        results_df["parallel_memory_MB"],
        marker="o",
        linewidth=2,
        label="Parallel CPU"
    )

    plt.xlabel("Number of Time Series (N)")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage Scaling")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


# -------------------------------------------------------
# Master Visualization Function
# -------------------------------------------------------
def generate_all_plots(results_df, C=None, output_dir="results", mode="cpu"):
    """
    Generate all standard plots (runtime, speedup, memory, correlation matrix).

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results DataFrame.
    C : np.ndarray or None
        Optional correlation matrix to visualize.
    output_dir : str
        Directory to save all generated plots.
    mode : str
        Execution mode prefix for plot filenames (e.g., 'cpu' or 'gpu').
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save plots with mode prefix to distinguish CPU vs GPU
    plot_runtime(results_df, f"{output_dir}/{mode}_runtime.png")
    plot_speedup(results_df, f"{output_dir}/{mode}_speedup.png")
    plot_memory(results_df, f"{output_dir}/{mode}_memory.png")

    if C is not None:
        plot_correlation_matrix(C, f"{output_dir}/{mode}_correlation_matrix.png")


# -------------------------------------------------------
# Standalone Testing Mode
# -------------------------------------------------------
if __name__ == "__main__":

    print("Running visualization demo...")

    # Fake benchmark data
    results = {
        "N": [500, 1000, 2000, 4000],
        "serial_time": [0.06, 0.18, 0.75, 3.0],
        "parallel_time": [0.20, 0.30, 1.10, 3.5],
        "speedup": [0.3, 0.6, 0.68, 0.85],
        "serial_memory_MB": [95, 120, 180, 350],
        "parallel_memory_MB": [97, 125, 190, 370]
    }

    df = pd.DataFrame(results)

    # Random correlation matrix demo
    C = np.random.randn(50, 50)

    generate_all_plots(df, C)

