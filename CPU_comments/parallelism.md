# Parallelism in the CPU Experiment Pipeline

This section explains how parallelism is used in the CPU experiment pipeline, focusing only on the execution structure of the experiment rather than the internal implementation details of helper functions.

---

# 1. Role of `run_cpu_experiments()`

The function `run_cpu_experiments()` acts as the **experiment controller**.  
It manages benchmarking across different dataset sizes.

The core structure is:

```python
for N in N_values:
    result = run_benchmark(N, T, num_workers)
```

Execution order:

```
N = 500   → run benchmark
N = 1000  → run benchmark
N = 2000  → run benchmark
N = 4000  → run benchmark
N = 8000  → run benchmark
N = 10000 → run benchmark
```

Each benchmark runs **sequentially**.

This is intentional because benchmark experiments must avoid resource interference that could distort runtime measurements.

Running experiments simultaneously would introduce:

- CPU contention
- Memory contention
- Inaccurate timing measurements

Therefore the **experiment driver remains sequential**.

---

# 2. Where Parallelism Actually Occurs

Parallelism occurs inside the **correlation matrix computation** used during benchmarking.

The correlation matrix is defined as:

\[
C = \text{corr}(X)
\]

where:

- \(X\) is the dataset of shape \(N \times T\)
- \(N\) = number of time series
- \(T\) = length of each time series

After standardization of the dataset:

\[
Z = \frac{X - \mu}{\sigma}
\]

the correlation matrix can be computed using matrix multiplication:

\[
C = \frac{Z Z^T}{T - 1}
\]

This transforms the problem into a **large linear algebra operation**.

---

# 3. Computational Complexity

The resulting correlation matrix has shape:

\[
C \in \mathbb{R}^{N \times N}
\]

The total number of correlations grows as:

\[
O(N^2)
\]

This quadratic growth is the main reason the computation becomes expensive for large datasets.

---

# 4. Parallelization Strategy Used

Instead of computing individual pairwise correlations, the implementation **parallelizes rows of the matrix multiplication**.

Each worker computes a subset of rows from the product:

\[
Z Z^T
\]

Example with **4 workers**:

| Worker | Rows Computed |
|------|------|
| Worker 1 | rows 0–249 |
| Worker 2 | rows 250–499 |
| Worker 3 | rows 500–749 |
| Worker 4 | rows 750–999 |

Each worker performs:

```
Zi @ Z.T
```

where:

```
Zi = subset of rows from Z
```

The result from each worker has shape:

```
(rows_i × N)
```

These partial matrices are then concatenated to form the final correlation matrix.

Conceptually:

```
Process 1 → compute C[0:250, :]
Process 2 → compute C[250:500, :]
Process 3 → compute C[500:750, :]
Process 4 → compute C[750:1000, :]
```

Final matrix:

```
C = stack(worker results)
```

---

# 5. Block-wise Computation Strategy

A second CPU strategy uses **block-wise computation**.

Instead of computing the full matrix multiplication at once, the computation is divided into **tiles (blocks)**:

\[
C_{ij} = \frac{Z_i Z_j^T}{T - 1}
\]

where:

- \(Z_i\) and \(Z_j\) represent subsets of rows from \(Z\)

Conceptually:

```
Z divided into blocks

Block 1 vs Block 1
Block 1 vs Block 2
Block 2 vs Block 1
Block 2 vs Block 2
```

This produces the full matrix through **many smaller matrix multiplications**.

### Advantages

- Reduced peak memory usage
- Scalable to very large datasets
- Compatible with GPU-style tiling

However, **smaller matrix multiplications reduce the efficiency of BLAS optimizations**.

---

# 6. BLAS Multithreading and Its Effect

NumPy internally uses BLAS libraries such as **OpenBLAS** or **MKL**.

These libraries already implement **highly optimized parallel linear algebra routines**.

In the default configuration:

```
NumPy matrix multiplication already uses multiple CPU threads
```

Therefore the **"serial" implementation is not truly single-threaded**.

Execution structure:

```
Python (single thread)
      ↓
BLAS library (multi-threaded)
```

This explains why the serial implementation often performs better than **Python-level multiprocessing**.

---

# 7. Experiment Configuration

To isolate the effect of Python multiprocessing, experiments were conducted in two configurations.

## Default BLAS Behavior

BLAS is allowed to use multiple threads.

This produces extremely fast matrix multiplication but reduces the benefit of Python multiprocessing.

## Forced Single-threaded BLAS

Environment variables were used to restrict BLAS to one thread:

```bash
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

This removes internal BLAS parallelism and allows a more direct comparison between:

- Python multiprocessing
- Single-thread CPU execution

---

# 8. Architecture of the CPU Experiment Pipeline

The CPU experiment system has three conceptual layers:

```
run_cpu_experiments()
      |
      | sequential experiment loop
      |
run_benchmark()
      |
      | calls different implementations
      |
+-------------------------------+
| Serial CPU                    |
| Multiprocessing CPU           |
| Block-wise CPU                |
+-------------------------------+
      |
      |
Correlation Matrix Computation
      |
+-----------+-----------+-----------+
| Worker 1  | Worker 2  | Worker 3  |
+-----------+-----------+-----------+
      |
      |
combine partial results
      |
      |
Correlation Matrix C
```

---

# 9. Type of Parallelism Used

The CPU implementation primarily uses **data parallelism**.

The dataset \(Z\) is partitioned into **row blocks**, and each worker performs the same computation on different subsets of rows.

This differs from other forms of parallelism:

| Type | Used Here |
|------|------|
| Data Parallelism | ✔ |
| Task Parallelism | ✘ |
| Pipeline Parallelism | ✘ |

---

# 10. Key Takeaway

The CPU experiment demonstrates an important practical lesson:

**Highly optimized numerical libraries can outperform naive parallelization strategies.**

The serial implementation benefits from:

- SIMD vectorization
- Cache optimization
- Low-level memory management
- Internal multithreading

As a result, **Python multiprocessing only becomes beneficial for very large problem sizes where BLAS optimizations alone are insufficient**.

