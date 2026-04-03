# GPU Correlation Matrix and Benchmark Technique

This repository contains a Python script designed to compute large-scale $N \times N$ correlation matrices for $N$ parallel time series. It serves as a benchmarking tool to demonstrate the performance and memory usage of GPU using PyTorch.

This was developed for **DS3294 - DS Practice Project #9**.

---

##  Project Overview

Computing a correlation matrix for thousands of variables (time series) requires millions of floating-point operations. As the number of time series ($N$) grows, the computational cost scales as $O(N^2 \cdot T)$, where $T$ is the number of time steps. 

This project tackles this problem by implementing two distinct computational strategies:
1. **GPU Full-Matrix:** Uses PyTorch to compute the entire matrix in one parallelized operation.
2. **GPU Blockwise:** A memory-efficient PyTorch implementation that processes the matrix in chunks to prevent VRAM overflow on large datasets.

---

##  The Original Code (Base Functionality)

The initial version of this script focused purely on building the PyTorch GPU logic and ensuring numerical accuracy. Its core functionalities included:

* **Synthetic Data Generation:** Generates a dataset of $N$ random time series of length $T$ using NumPy.
* **Device Agnosticism:** Automatically detects if an NVIDIA CUDA GPU is available; otherwise, it falls back to the CPU.
* **Strategy A (Full GPU):** * Transfers data to the GPU.
  * Standardizes the data (Z-scoring).
  * Computes the Pearson correlation matrix via matrix multiplication (`torch.mm`).
  * **Pros:** Blazing fast. **Cons:** Can exceed GPU VRAM for very large $N$.
* **Strategy B (Blockwise GPU):**
  * Divides the $N \times N$ matrix calculation into smaller sub-blocks (e.g., $512 \times 512$).
  * Computes correlations block-by-block and stitches them together.
  * **Pros:** Strict upper bound on VRAM usage. **Cons:** Slightly slower than Strategy A.
* **Numerical Verification:** Cross-checks the results of the PyTorch mathematical operations against NumPy's standard `np.corrcoef` to guarantee accuracy.

---

##  Gemini Alterations & New Features

The code was subsequently updated to transform it from a pure mathematical script into a complete benchmarking and profiling tool. The following functionalities were added:

### 1. Comprehensive Memory Tracking
* **GPU VRAM Profiling:** Integrated `torch.cuda.max_memory_allocated()` and `torch.cuda.reset_peak_memory_stats()` to capture the exact peak VRAM footprint of both PyTorch strategies.

### 2. Automated Data Visualization
* **Matplotlib Dashboard:** Added a `plot_comparisons` function that automatically generates a side-by-side bar chart at the end of the experiment.
* **Visual Insights:** The charts visually compare the **Execution Time (seconds)** and **Peak Memory Usage (MB)** across the two methods, making the hardware performance differences immediately clear.

---

## How to Run

### Prerequisites
Ensure you have a Python environment set up with the following libraries:
* `torch` (PyTorch with CUDA support recommended)
* `numpy`
* `matplotlib`

### Execution
Run the script directly from your terminal or IDE (like VS Code):
```bash
python gpu_benchmark.py
