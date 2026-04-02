# Parallel Data Processing Engine

A high-performance benchmarking suite designed to evaluate and compare data processing efficiency across **CPU** (Serial, Parallel, Block-wise) and **GPU** architectures.  
The project focuses on identifying the **break-even point** where computational gains outweigh memory transfer overheads.

---

## 📂 Repository Structure

```
parallel_data_processing_engine/
├── run_experiment.py          # Main entry point (CPU + GPU unified CLI)
├── results/                   # Auto-generated experiment outputs
│
└── src/
    ├── dataset.py             # Shared: data loading + synthetic generation
│
    ├── cpu/                   # CPU-based implementations
    │   ├── __init__.py
    │   ├── serial_cpu.py      # Single-threaded correlation
    │   ├── parallel_cpu.py   # Multi-threaded implementation
    │   ├── block_cpu.py      # Cache-optimized block computation
    │   ├── benchmark.py      # CPU benchmarking logic
    │   └── visualize.py      # CPU plotting (runtime, speedup, memory, corr)
│
    └── gpu/                   # GPU-based implementations
        ├── __init__.py
        ├── gpu_correlation.py   # Full + blockwise GPU correlation
        ├── gpu_benchmark.py     # GPU benchmarking logic
        └── gpu_visualize.py     # GPU plotting (runtime, memory, corr)

```

---

## ⚡ Running Experiments

The engine uses a **CLI-based approach** to toggle between different optimization levels and BLAS (Basic Linear Algebra Subprograms) configurations.

### CPU Benchmarks

| Configuration | Command |
|---------------|---------|
| Baseline (Single-thread BLAS) | `python run_experiment.py --mode cpu --version baseline --blas single` |
| Optimized (Single-thread BLAS) | `python run_experiment.py --mode cpu --version optimized --blas single` |
| Baseline (Multi-thread BLAS) | `python run_experiment.py --mode cpu --version baseline --blas multi` |
| Optimized (Multi-thread BLAS) | `python run_experiment.py --mode cpu --version optimized --blas multi` |
| If real data is to be used add the following line (with proper data path and file name) and data id. And please follow the dataformat | `--dataset real --data_path src/data/global_temp_use.csv --data_id global_temp`|


### GPU Benchmarks (Planned)

> **Note:**  
> GPU: Sipra is working on it  
> Current GPU script is in [GPU Parallel Processing Module](https://github.com/AmitavaDutta/parallel_data_processing_engine/tree/main/GPU_parallelproce)  
> In case of GPU, if a single code structure is implemented as CPU, do the following:
```bash
python run_experiment.py --mode gpu
```

> All results are automatically timestamped and saved in the `results/` directory.  
> (For now only the CPU execution part is pipelined that way)

---
### Testing the implementation on an actual dataset
The initial benchmarking was performed on randomly generated dataset. We will use temperature data over a decade from a large number of locations (N) obtained from NASA POWER API. This is to ensure that the implementation works on actual datasets too.  
## 🛠 Project Workflow & Responsibilities

| Contributor | Project Phase |
|-------------|---------------|
| **Amitava** | CPU Implementation (Single/Multi-thread) | 
|             | Block-wise Computation (CPU) | 
|             | Theoritical Complexity Analysis ($O(N^2)$) for CPU|
| **Sipra** | GPU Implementation | 
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

---

# Presentation

[View the Presentation](https://docs.google.com/presentation/d/1Owx46RyIDORviqmCtD00_zH3qimb5DthErWeH7hPy30/edit?usp=sharing)
