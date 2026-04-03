# Parallel Data Processing Engine

**Team 13** **Members:** Amitava, Sipra, Bhavini, Yashvita

---

## Project Description

This repository contains a high-performance benchmarking suite designed to evaluate and compare data processing efficiency across CPU (Serial, Parallel, Block-wise) and GPU architectures. Using correlation matrix estimation from large collections of time series as a case study, the project systematically analyzes $O(N^2)$ complexity bottlenecks. The primary objective is to identify the break-even points where hardware computational gains successfully outweigh data serialization and memory transfer overheads as data dimensionality scales.

The testing encompasses both synthetic random data and real-world empirical datasets—specifically NASA POWER temperature data and financial time series—to ensure robust performance analysis under varying spatial and temporal constraints.

---

## Repository Structure

The repository is organized into distinct functional domains, separating the unified benchmarking engine from specialized exploratory analyses and validation suites.

```text
parallel_data_processing_engine/
│
├── README.md                      # Project documentation and execution guide
├── LICENSE                        # Project license and usage terms
│
├── Core Engine & Execution
│   ├── run_experiment.py          # Unified CLI entry point for all benchmarks
│   ├── src/                       # Primary source code modules
│   │   ├── dataset.py             # Shared data ingestion and synthetic generation
│   │   ├── cpu/                   # CPU correlation pipelines (Serial, Parallel, Block)
│   │   │   ├── serial_cpu.py      # Symmetry-aware single-threaded logic (50% FLOP reduction)
│   │   │   ├── parallel_cpu.py    # Multi-process implementation utilizing shared-memory buffers
│   │   │   ├── block_cpu.py       # Tiling strategy optimized for L3 cache hit-rates
│   │   │   ├── benchmark.py       # Automated wall-clock and RSS memory profiling suite
│   │   │   └── visualize.py       # Performance curve and correlation heatmap generation
│   │   ├── gpu/                   # CUDA-accelerated correlation pipelines
│   │   │   ├── gpu_correlation.py # CUDA-accelerated tensor operations with VRAM-aware tiling
│   │   │   ├── gpu_benchmark.py   # High-precision profiling of kernel latency and VRAM occupancy
│   │   │   └── gpu_visualize.py   # GPU performance scaling and numerical consistency validation 
│   │   └── data/                  # Source datasets for empirical validation
│   │       ├── data_format_readme.md
│   │       └── global_climate_master.csv
│   └── results/                   # Experiment outputs (CSV & Plots)
│
└── Research Artifacts & Support   # These are the rest of the files and directories that
    │                                 contain some tests, some datasets
    └── [Experimental Files]       # Supplementary notebooks, standalone tests, and docs
```

---

## Execution Guide

The engine provides a unified Command Line Interface (CLI) via `run_experiment.py` for executing standard benchmarks. If no dataset is explicitly passed, the engine defaults to generating a synthetic random dataset ($N \times T$) for baseline testing.

---

### CPU Benchmarks

To execute CPU experiments, specify the algorithm `--version` (`baseline` or `optimized`) and the `--blas` threading backend (`single` or `multi`).

**Basic Usage:**
```bash
python run_experiment.py --mode cpu --version <baseline|optimized> --blas <single|multi>
```

**Using Real-World Dataset:**
```bash
python run_experiment.py --mode cpu --version optimized --blas multi --dataset real --data_path src/data/global_temp_use.csv --data_id global_temp
```

---

### GPU Benchmarks

GPU execution follows an identical interface. Note that BLAS threading control is not required as it is natively managed by the CUDA backend.

**Basic Usage:**
```bash
python run_experiment.py --mode gpu --version <baseline|optimized>
```

**Using Real-World Dataset:**
```bash
python run_experiment.py --mode gpu --version optimized --dataset real --data_path src/data/global_temp_use.csv --data_id global_temp
```

---

## Hardware Detection & Safety

- **CUDA Detection:** The engine automatically scans for available CUDA-compatible hardware. Upon success, the system initializes the GPU device:
  
  ```
  [Device] CUDA-compatible GPU detected.
  ```

- **Graceful Exit:** In environments where a GPU is requested but not detected, the program terminates safely without data loss.

---

## Data Management & Output

### Dataset Specifications

- Initial benchmarking is conducted using synthetic matrices to establish theoretical complexity bounds.
- For empirical validation, the engine utilizes structured real datasets.

**Format:**
- Input matrices follow an **N × T** structure  
  - **N** = number of locations  
  - **T** = number of time steps  

**Validation:**
- Ensures numerical consistency and performance stability on structured datasets.

---

### Output Hierarchy

All experimental results—including runtime CSV logs, memory profiles, and correlation outputs—are automatically routed to the `results/` directory using a hierarchical structure:

```text
results/
├── cpu/
│   └── <version>/<dataset_id>/...
└── gpu/
    └── <version>/<dataset_id>/...
```

**Example Output Path:**
```
results/gpu/optimized/global_temp/results_global_temp_optimized_gpu.csv
```

### Testing the implementation on an actual dataset
The initial benchmarking was performed on randomly generated dataset. We will use temperature data over a decade from a large number of locations (N) obtained from NASA POWER API. This is to ensure that the implementation works on actual datasets too.  
## 🛠 Project Workflow & Responsibilities

| Contributor | Project Phase |
|-------------|---------------|
| **Amitava** | Created and made subsequent editing of the main README.md|
|             | CPU Implementation (Single/Multi-thread) | 
|             | Block-wise Computation (CPU) | 
|             | Theoritical Complexity Analysis ($O(N^2)$) for CPU|
|             | Created tge repository structure and implemented various .py modules in src/ so as to smooth working of run_experiment.py|
|             | Integrated GPU scripts into run_experiment.py, and from the functions implemented them into .py modules |
|             | Written and edited the report (shared)|
| **Sipra**   | GPU Implementation | 
|             | Block-wise Computation (GPU) | 
| **Bhavini** | Numerical Consistency Check (CPU vs GPU) on Financial Dataset|
|             | Complexity Analysis ($O(N^2)$) on Financial dataset|
|             | System Profiling (CPU/GPU/RAM bottlenecks) on Financial Dataset| 
|             | Experiment using real world data Execution & Plotting (Shared) on Financial Dataset | 
| **Yashvita** | Bandwidth & Data Transfer Analysis: Pinned vs Paged memory | 
|             | Transfer Overhead Optimization | 
|             | Try out the implementation on climate dataset | 
|             | Written and edited the report (shared)|


---
