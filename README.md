# Parallel Data Processing Engine

A high-performance benchmarking suite designed to evaluate and compare data processing efficiency across **CPU** (Serial, Parallel, Block-wise) and **GPU** architectures.  
The project focuses on identifying the **break-even point** where computational gains outweigh memory transfer overheads.

---

## 📂 Repository Structure

```
parallel_data_processing_engine/
├── run_experiment.py         # Main entry point for benchmarking
├── results/                  # Auto-generated experiment outputs
└── src/
    ├── cpu/                  # CPU-based implementations
    │   ├── __init__.py
    │   ├── dataset.py        # Data loading and synthetic generation
    │   ├── serial_cpu.py     # Single-threaded implementation
    │   ├── parallel_cpu.py   # Multi-threaded implementation
    │   ├── block_cpu.py      # Cache-optimized block computation
    │   ├── benchmark.py      # CPU performance metrics
    │   └── visualize.py      # Plotting and results analysis
    │
    └── gpu/                  # GPU-based implementations (WIP)
        └── __init__.py       # Future CUDA/OpenCL implementations
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

## 🛠 Project Workflow & Responsibilities

| Contributor | Project Phase |
|-------------|---------------|
| **Amitava** | CPU Implementation (Single/Multi-thread) | 
|             | Block-wise Computation (Shared) | 
| **Sipra** | GPU Implementation | 
|             | Block-wise Computation (Shared) | 
| **Bhavini** | Numerical Consistency Check (CPU vs GPU) |
|             | Complexity Analysis ($O(N^2)$) |
|             | Overhead vs Computation Equilibrium | 
|             | System Profiling (CPU/GPU/RAM bottlenecks) | 
|             | Code Revision & Optimization | 
|             | Experiment Execution & Plotting (Shared) | 
| **Yashvita** | Bandwidth & Data Transfer Analysis | 
|             | Transfer Overhead Optimization | 
|             | Experiment Execution & Plotting (Shared) | 

---

## 🔬 Methodology

The core focus is **empirical analysis of $O(N^2)$ operations**. The project objectives are:

1. **Identify Bottlenecks**  
   Use system profiling to determine if the performance lag is due to compute logic or RAM bandwidth.

2. **Optimize Transfers**  
   Implement strategies to minimize latency between **Host (CPU)** and **Device (GPU)** memory.

3. **Validate Accuracy**  
   Ensure that optimized **Block-wise** approaches maintain **bit-perfect numerical consistency** with the baseline serial code.

# Presentation

[View the Presentation](https://docs.google.com/presentation/d/1Owx46RyIDORviqmCtD00_zH3qimb5DthErWeH7hPy30/edit?usp=sharing)
