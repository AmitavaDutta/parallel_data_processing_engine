# Parallelism in the CPU Experiment Pipeline

This section explains how parallelism is used in the CPU experiment pipeline, focusing on the execution structure, the shared memory architecture, and the benchmarking mechanics.

---

# 1. Role of `run_cpu_experiments()`

The function `run_cpu_experiments()` acts as the **experiment controller**. It manages systematic benchmarking across various dataset sizes \((N)\).

Core structure:

```python
for N in N_values:
    result = run_benchmark(N, T, num_workers)
```

Execution order:

```
N = 500 → N = 1000 → N = 2000 → ... → N = 20000
```

## Sequential Constraint

Each benchmark runs **sequentially**.

This is a deliberate design choice for scientific benchmarking. Running multiple experiments simultaneously would introduce:

- CPU contention  
- Memory contention  
- Noisy measurements  

The experiment driver remains sequential to ensure **accurate and reproducible timing results**.

---

# 2. Where Parallelism Actually Occurs

Parallelism occurs inside the **correlation matrix computation**.

After standardizing the dataset:

\[
Z = \frac{X - \mu}{\sigma}
\]

the correlation matrix is computed as:

\[
C = \frac{Z Z^T}{T - 1}
\]

This transforms the problem into a **large linear algebra operation** with complexity:

\[
O(N^2)
\]

---

# 3. Parallelization Strategy: Shared Memory Model

The implementation uses **data parallelism with shared memory** to reduce overhead.

---

## The Zero-Copy Approach

Instead of copying data to each worker:

- **Allocation:** Parent process allocates shared memory for \(Z\)  
- **Access:** Workers receive a reference (shared memory name)  
- **Execution:** All workers read the same physical memory  

This eliminates:

- Pickling  
- Data duplication  
- IPC overhead  

---

## Row-wise Distribution

The rows of the correlation matrix are partitioned across workers:

| Worker | Responsibility | Computation |
|------|------|------|
| Worker 1 | Rows \(0\) to \(i\) | \(Z_{chunk1} \times Z^T\) |
| Worker 2 | Rows \(i+1\) to \(j\) | \(Z_{chunk2} \times Z^T\) |
| ... | ... | ... |

Each worker computes a partial result:

\[
\text{Partial}_i = Z_{chunk} \times Z^T
\]

Final matrix assembly:

```
C = vstack(worker_results)
```

---

# 4. Block-wise (Tiled) Strategy

A secondary strategy uses **block-wise computation**:

\[
C_{ij} = \frac{Z_i Z_j^T}{T - 1}
\]

## Memory Mapping

- Uses `np.memmap` to store intermediate results on disk  
- Keeps RAM usage low  

## Trade-off

- Nested loops introduce overhead  
- Smaller matrix multiplications reduce BLAS efficiency  

---

# 5. BLAS Multithreading vs. Explicit Parallelism

BLAS plays a critical role in performance.

---

## Implicit Parallelism (Default)

- NumPy uses BLAS (OpenBLAS/MKL)  
- Automatically utilizes multiple CPU threads  

Result:

```
"Serial" implementation is already parallel
```

---

## Explicit Parallelism (Controlled)

To isolate multiprocessing performance:

```bash
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

This ensures:

- One BLAS thread per worker  
- No oversubscription  

---

# 6. Architecture of the Experiment Pipeline

```
[ Experiment Controller: run_cpu_experiments ]
             |
             | (Sequential Loop over N)
             v
[ Benchmark Engine: run_benchmark ]
             |
             |--> [ Recursive Memory Tracker ]
             |    (Parent + All Child Processes)
             |
             |--> [ Implementation Selection ]
                  |-- Serial (Single-thread BLAS)
                  |-- Parallel (Shared Memory Workers)
                  |-- Block-wise (Tiled / Memmap)
             v
[ Result Aggregator ] → CSV / Visualization
```

---

# 7. Recursive Memory Tracking

Standard memory tracking is insufficient.

This pipeline uses **recursive tracking**:

1. Record memory before execution  
2. Spawn worker processes  
3. Traverse all child processes  
4. Sum their **Resident Set Size (RSS)**  

Result:

- Accurate measurement of total memory usage  
- Captures full parallel footprint  

---

# 8. Key Takeaways

- **Overhead Matters:**  
  Even with shared memory, process creation cost makes serial BLAS faster for small datasets.

- **Efficiency Threshold Exists:**  
  There is a dataset size \(N\) where multiprocessing begins to outperform BLAS.

- **Architecture is Critical:**  
  Transitioning from data copying to shared memory is essential for scaling large problems.

---

# Final Insight

Efficient parallel computing depends on:

```
problem size + memory architecture + hardware utilization
```

Not just the presence of parallelism.

