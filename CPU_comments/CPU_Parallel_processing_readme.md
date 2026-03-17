# Analysis of CPU Implementations for Correlation Matrix Computation

We benchmarked three CPU-based implementations for computing the correlation matrix of \(N\) time series of length \(T\):

- **Serial CPU (NumPy):** Baseline using standard dot products.  
- **Multiprocessing CPU (Shared Memory):** Parallelized using Python’s multiprocessing with a shared memory buffer to avoid IPC overhead.  
- **Block-wise CPU (Tiled):** Memory-efficient approach using smaller matrix tiles.  

Experiments were performed under two configurations:

- **BLAS Multithreaded:** Default NumPy behavior (implicit parallelism).  
- **BLAS Single-threaded:** Forcing NumPy to one thread (explicit parallelism control).  

---

# Performance Dynamics of Parallel vs. Serial Implementations

While we optimized the parallel implementation using shared memory, the serial implementation often remains the most efficient for moderate \(N\).

---

## 1. Optimized BLAS vs. Python Orchestration

The serial implementation computes the matrix via:

\[
C = \frac{Z Z^T}{T - 1}
\]

NumPy delegates this to BLAS (OpenBLAS/MKL). These libraries are written in C/Fortran and optimized at the hardware level for:

- SIMD vectorization  
- Cache locality  
- Multithreading  

Because the serial version passes one large matrix to BLAS, the library can maximize its internal threading and vectorization efficiency.

---

## 2. Mitigating but Not Eliminating Parallel Overhead

In the updated implementation, we used `multiprocessing.shared_memory`.

### IPC Improvement

- Shared memory eliminates the need to pickle and copy the \(Z\) matrix to every worker.  
- This significantly improves runtime compared to standard multiprocessing.

### Persistent Overheads

Despite shared memory, key overheads remain:

- **Process Creation Overhead:** Spawning and initializing 4–8 worker processes is expensive.  
- For smaller \(N\), this setup cost exceeds the benefit of parallel computation.

---

## 3. The Efficiency Crossing Point

The **parallel CPU implementation becomes beneficial only at larger \(N\)**.

- BLAS is highly optimized but eventually faces limits with very large memory workloads.  
- Explicit multiprocessing enables manual load balancing across cores.  

As \(N\) increases beyond cache-friendly sizes:

- Computation dominates overhead  
- Parallel execution becomes competitive  

---

# Behavior of the Block-wise Implementation

The block-wise approach divides the computation into tiles:

\[
C_{block} = \frac{Z_i Z_j^T}{T - 1}
\]

---

## Memory-Constraint Strategy

In the updated `block_cpu.py`, memory mapping (`memmap`) was introduced:

- Enables computation for \(N = 20{,}000+\)  
- Uses disk as an extension of RAM  
- Prevents memory overflow  

---

## Performance Trade-offs

- **Loop Overhead:** Each block introduces Python-level loop cost.  
- **Reduced BLAS Utilization:** Smaller tiles prevent BLAS from applying full optimization.  

However:

- This approach is the most **stable**  
- Prevents crashes in high-dimensional runs  

---

# BLAS Threading and Resource Contention

A critical issue observed is **oversubscription**.

---

## 1. BLAS Multithreaded (Default)

- NumPy already uses multiple threads internally.  

If multiprocessing is added:

```
4 processes × 4 BLAS threads = 16 threads
```

On a system with fewer cores:

- Threads compete for CPU  
- Context switching increases  
- Performance degrades  

---

## 2. BLAS Single-threaded (Forced)

By setting:

```bash
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

- Each worker uses exactly one core  
- Eliminates oversubscription  
- Provides clean scaling behavior  

---

# Accurate Memory Usage Observations

With recursive memory tracking:

---

## Parent + Children Accounting

- Total memory now includes all worker processes  
- Parallel implementation shows significantly higher usage than serial  

---

## Shared Memory Footprint

- Shared memory reduces duplication  
- However, OS still accounts memory in each process’s virtual space  

---

## Block-wise Memory Behavior

- Most memory-efficient approach  
- Only small tiles remain in RAM at any time  
- Suitable for very large \(N\)  

---

# Summary of Implementation Characteristics

| Method | Optimization Level | Best Use Case |
|------|------|------|
| **Serial CPU** | High (Internal BLAS) | Moderate \(N\), fastest baseline |
| **Multiprocessing (Shared Memory)** | Medium (Custom) | Large \(N\), when BLAS scaling saturates |
| **Block-wise CPU** | Memory-focused | Very large \(N\), limited RAM |

---

# Key Conclusion

Parallelism is **not inherently beneficial**.

Even with shared memory optimization:

- Process management overhead remains significant  
- Performance depends on problem size  

---

## Problem Regime Summary

- **Small \(N\):** Serial (BLAS) is optimal due to zero overhead  
- **Large \(N\):** Multiprocessing becomes competitive as compute dominates overhead  
- **Extreme \(N\):** Block-wise with memmap is required to avoid memory failures  

---

**Final Insight:**  
The optimal strategy is determined by the **scale of the problem and system constraints**, not just the presence of parallelism.

