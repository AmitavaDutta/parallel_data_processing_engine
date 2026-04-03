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
│   │   ├── cpu/                   # CPU correlation pipelines
│   │   │   ├── __init__.py
│   │   │   ├── serial_cpu.py      # Single-threaded logic and symmetry routing
│   │   │   ├── parallel_cpu.py    # Multi-threaded multiprocessing implementation
│   │   │   ├── block_cpu.py       # Cache-optimized block computation
│   │   │   ├── benchmark.py       # CPU execution timing and memory profiling
│   │   │   └── visualize.py       # CPU plotting (runtime, speedup, correlation)
│   │   └── gpu/                   # CUDA-accelerated correlation pipelines
│   │       ├── __init__.py
│   │       ├── gpu_correlation.py # Full-matrix and block-wise tensor operations
│   │       ├── gpu_benchmark.py   # VRAM profiling and GPU execution timing
│   │       └── gpu_visualize.py   # GPU plotting and consistency validation
│   └── results/
│           ├── cpu/
│           │   └── <version>/<dataset_id>/...
│           └── gpu/
│               └── <version>/<dataset_id>/...                   
│
├── Exploratory Analysis & Notebooks
│   ├── Bandwidth.ipynb            # Empirical analysis of memory transfer overhead
│   ├── GPU.ipynb                  # Interactive environment for isolated GPU testing
│   └── GPU_versions.ipynb         # Comparative study of block-wise vs. full-matrix logic
│
├── Specialized Benchmarks & Auxiliary Scripts
│   ├── run_experiment_cpu.py      # Standalone execution wrapper for isolated CPU testing
│   ├── run_experiment_gpu.py      # Standalone execution wrapper for isolated GPU testing
│   ├── timing_comparison.py       # Micro-benchmarking script for targeted operations
│   ├── test_cases/                # Unit tests and numerical validation suites (CPU vs GPU)
│   ├── financial_time_series/     # Experimental environment for sector-specific datasets
│   └── GPU_parallelproce/         # Prototype parallel processing routines for CUDA
│
└── Documentation & Project Resources
    ├── Project_descriptions/      # Original academic guidelines and architectural drafts
    ├── GPU_CPU.docx               # Theoretical complexity analysis and architectural notes
    └── CPU_comments/              # Code review logs and CPU optimization strategies
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
|             | Integrated Sipra's GPU scripts into run_experiment.py, and from the functions written by Sipra implemented them into .py modules |
|             | Written and edited the report (shared)|
| **Sipra**   | GPU Implementation | 
|             | Block-wise Computation (GPU) | 
| **Bhavini** | Numerical Consistency Check (CPU vs GPU) |
|             | Overhead vs Computation Equilibrium | 
|             | Complexity Analysis ($O(N^2)$) |
|             | System Profiling (CPU/GPU/RAM bottlenecks) | 
|             | Code Revision & Optimization | 
|             | Experiment using real world data Execution & Plotting (Shared) | 
| **Yashvita** | Bandwidth & Data Transfer Analysis | 
|             | Transfer Overhead Optimization | 
|             | Try out the implementation on climate dataset | 
|             | Written and edited the report (shared)|


---
