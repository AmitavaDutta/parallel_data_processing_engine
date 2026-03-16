# parallel_data_processing_engine
### Ideal Project Structure
```
correlation-parallel-project/

GPU_CPU.docx
LICENSE
Project_descriptions
README.md
test_cases/

src/
    cpu/
        dataset.py
        serial_cpu.py
        parallel_cpu.py
        benchmark.py
        visualize.py


    gpu/        <-- for the other person later
        #gpu_[...].py whatever is required

run_experiment.py
results/

```
## 👥 Contributors

[![Contributors](https://contrib.rocks/image?repo=AmitavaDutta/parallel_data_processing_engine)](https://github.com/AmitavaDutta/parallel_data_processing_engine/graphs/contributors)

## 📊 Contributor Activity

<table>
<tr>
<td align="center">

### AmitavaDutta
<img src="https://github-readme-activity-graph.vercel.app/graph?username=AmitavaDutta"/>

</td>
<td align="center">

### S Yashvita
<img src="https://github-readme-activity-graph.vercel.app/graph?username=SYashvita"/>

</td>
</tr>

<tr>
<td align="center">

### Sipra Subhadarsini Sahoo
<img src="https://github-readme-activity-graph.vercel.app/graph?username=Sipra-S"/>

</td>
<td align="center">

### Bhavini
<img src="https://github-readme-activity-graph.vercel.app/graph?username=bhaviniraina"/>

</td>
</tr>
</table>

![Contribution Snake](https://raw.githubusercontent.com/Platane/snk/output/github-contribution-grid-snake.svg)

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
