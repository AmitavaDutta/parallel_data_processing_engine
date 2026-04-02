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

## # ⚡ Running Experiments

The engine provides a **CLI-based interface** to run CPU and GPU benchmarks with flexible configurations.

---

## 🖥️ CPU Benchmarks

Run experiments by specifying:

- `--version` → `baseline` or `optimized`  
- `--blas` → `single` or `multi`  

---

### ▶️ Basic Usage

```bash
python run_experiment.py --mode cpu --version <baseline|optimized> --blas <single|multi>
```

---

### 📊 Using Real Dataset

```bash
python run_experiment.py --mode cpu --version optimized --blas multi --dataset real --data_path src/data/global_temp_use.csv --data_id global_temp
```

---

### 📌 Default Behavior

- If `--dataset` is **not specified**, a **synthetic random dataset** is generated automatically.

---

## 🚀 GPU Benchmarks

GPU execution follows the **same interface** (no BLAS control required).

---

### ▶️ Basic Usage

```bash
python run_experiment.py --mode gpu --version <baseline|optimized>
```

---

### 📊 Using Real Dataset

```bash
python run_experiment.py --mode gpu --version optimized --dataset real --data_path src/data/global_temp_use.csv --data_id global_temp
```

---

## 🧠 Notes

- GPU execution **automatically detects available CUDA devices**:

```
[Device] GPU detected: NVIDIA T1000 8GB (7.6 GB VRAM)
```

- If no GPU is available:
  - The program exits **gracefully**

---

## 📁 Output Structure

Results are automatically saved to:

```
results/cpu/...   # CPU runs
results/gpu/...   # GPU runs
```

### Example Output

```
results/gpu/optimized/global_temp/results_global_temp_optimized_gpu.csv
```

---

## 🧪 Dataset

- **Default:** Synthetic random dataset  
- **Real dataset:** NASA POWER temperature data  

### Format

- Matrix shape: **N × T**  
  - **N** = number of locations  
  - **T** = number of time steps  

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
