# parallel_data_processing_engine
### Ideal Project Structure
```
parallel_data_processing_engine/
├── run_experiment.py
├── results/
└── src/
    ├── __init__.py         <-- Create this (empty file)
    └── cpu/
        ├── __init__.py     <-- Create this (empty file)
        ├── dataset.py
        ├── serial_cpu.py
        ├── parallel_cpu.py
        ├── block_cpu.py
        ├── benchmark.py
        └── visualize.py


    gpu/        <-- for the other person later
        #gpu_[...].py whatever is required


```
To run the run_experiment.py
```
# Baseline, single-thread BLAS
python run_experiment.py --mode cpu --version baseline --blas single

# Optimized, single-thread BLAS
python run_experiment.py --mode cpu --version optimized --blas single

# Baseline, multi-thread BLAS
python run_experiment.py --mode cpu --version baseline --blas multi

# Optimized, multi-thread BLAS
python run_experiment.py --mode cpu --version optimized --blas multi

```

### GPU Benchmark Extension Plan
```
## 1. Dataset Generation
- Reuse `dataset.py` (synthetic data is sufficient).
- Optionally, allow passing the dataset to GPU as a `cupy` or `torch` tensor.

## 2. GPU Correlation Implementations (`src/gpu/gpu_correlation.py`)
- Implement at least **two strategies**:
  1. **Full correlation matrix computation**: compute the entire matrix at once (may hit memory limit for large N).
  2. **Block-wise / chunked computation**: split the matrix to avoid GPU memory overflow.
- Include optional **timing of data transfer** (CPU → GPU and GPU → CPU).

## 3. GPU Benchmarking
- Write `run_benchmark_gpu(N, T, strategy, seed)` (or extend existing `run_benchmark`) to:
  - Generate dataset.
  - Move data to GPU.
  - Measure time for:
    - CPU → GPU transfer
    - GPU computation
    - GPU → CPU transfer
  - Return runtime breakdown, memory usage, and correctness compared to CPU results.

## 4. Integration into `run_experiment.py`
- Extend `run_gpu_experiments()` to:
  - Loop over different `N` values.
  - Loop over different GPU strategies.
  - Call GPU benchmark for each combination.
  - Collect results in a DataFrame (similar to CPU).
  - Save CSV as `results_gpu.csv`.
  - Generate plots with `mode="gpu"`.

## 5. Memory-Limited Scenario
- For very large `N`, demonstrate **block-wise GPU computation**.
- Show that the full matrix may fail due to GPU memory.
- Compare performance and memory usage.

## 6. Visualization
- Reuse `visualize.py`.
- Pass `mode="gpu"` for filenames to avoid overwriting CPU plots.
- Plots show:
  - Runtime
  - Speedup vs CPU
  - Memory usage
  - Optional: correlation matrix heatmap

## 7. Profiling / Bottleneck Analysis
- Compare GPU strategies to show which is faster under different problem sizes.
- Measure data transfer overhead to highlight cases where CPU ↔ GPU communication dominates performance.
```
## Some important Instructions
```
Shared experiment runner

Keep this in root:

run_experiment.py


This script can run both:

CPU benchmark
GPU benchmark


Later you could even do:

python run_experiment.py --mode cpu ## default run:: python run_experiment.py
python run_experiment.py --mode gpu

Results folder

Create:

results/


Save:

runtime_plot.png
correlation_heatmap.png
benchmark_table.csv


This is very useful for the report.
```

### GPU computing can to be done in google collab. Everyone won't have a dedicated GPU and even if a dedicated GPU is available there are many dependencies required for CuPy or PyTorch that may take up quiet some time to install locally, so best idea would be to use google collab.
