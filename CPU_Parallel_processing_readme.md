## Analysis of CPU Implementations for Correlation Matrix Computation

We benchmarked three CPU-based implementations for computing the correlation matrix of (N) time series of length (T):

1. **Serial CPU (NumPy)**
2. **Multiprocessing CPU**
3. **Block-wise CPU**

Experiments were performed under two configurations:

* **BLAS Multithreaded (default NumPy behavior)**
* **BLAS Single-threaded (forcing NumPy to one thread)**

The results reveal several important performance characteristics related to **parallelism, overhead, and optimized numerical libraries**.

---

# Why Parallel Processing is Slower than the Serial Implementation

Although multiprocessing introduces parallelism at the Python level, the **serial implementation consistently performs faster in these experiments**. Several factors explain this behavior.

## 1. Highly Optimized BLAS Operations in NumPy

The serial implementation computes the correlation matrix primarily through the matrix multiplication

[
C = \frac{ZZ^T}{T-1}
]

where (Z) is the standardized data matrix.

NumPy internally delegates this operation to **BLAS (Basic Linear Algebra Subprograms)** libraries such as **OpenBLAS** or **MKL**, which are heavily optimized for:

* SIMD vectorization
* CPU cache utilization
* Multithreaded execution
* Low-level memory management

As a result, even the "serial" Python code actually executes a **highly optimized native implementation**, making it extremely efficient.

---

## 2. Overhead of Multiprocessing

The multiprocessing implementation divides the work across multiple processes. However, this introduces significant overhead:

### Process Creation

Each worker process must be spawned and initialized, which is computationally expensive relative to the workload for moderate dataset sizes.

### Inter-process Communication

Large data arrays must be **copied or serialized** between processes. Since Python processes do not share memory by default, this results in additional memory allocation and copying costs.

### Memory Duplication

Each process may hold its own copy of the dataset or intermediate arrays, leading to:

* Increased memory usage
* Reduced cache locality
* Higher memory bandwidth pressure

These overheads often dominate the computation time for moderately sized problems.

---

## 3. Loss of BLAS-Level Optimization

When using multiprocessing, the computation is often broken into smaller chunks. These smaller matrix operations do not utilize BLAS optimizations as efficiently as the large matrix multiplication used in the serial implementation.

As a result:

* SIMD efficiency decreases
* CPU cache utilization becomes worse
* BLAS cannot fully exploit vectorized operations

This further reduces performance relative to the serial implementation.

---

# Behavior of the Block-wise Implementation

The **block-wise approach** computes the correlation matrix in smaller tiles rather than computing the entire matrix at once.

Conceptually:

[
C_{ij} = \frac{Z_i Z_j^T}{T-1}
]

where (Z_i) and (Z_j) represent blocks of rows from the standardized matrix.

## Advantages

The block-wise strategy is useful for:

* **Reducing peak memory usage**
* Handling **very large datasets**
* Enabling **GPU-style tiling strategies**

It is commonly used in high-performance computing when the full matrix does not fit in memory.

## Performance Observations

However, in the current experiments the block-wise implementation is slower than the serial implementation because:

### Increased Loop Overhead

The algorithm introduces nested loops over blocks, which adds Python-level control overhead.

### Smaller Matrix Multiplications

Each block multiplication is smaller, which reduces the efficiency of BLAS optimizations.

Large matrix operations allow BLAS libraries to:

* better utilize SIMD
* improve cache reuse
* amortize memory access costs

Breaking the computation into smaller pieces reduces these benefits.

### Additional Memory Operations

Intermediate blocks must be repeatedly allocated and written back to the final matrix, increasing memory traffic.

---

# Effect of BLAS Multithreading vs Single-threaded BLAS

Two configurations were tested:

### 1. BLAS Multithreaded (Default NumPy)

In this configuration, NumPy's BLAS backend automatically uses multiple CPU threads for matrix operations.

This means the **serial implementation is already parallelized internally**.

As a result:

* Serial computation becomes extremely fast.
* Python-level multiprocessing provides little benefit.
* Multiprocessing may even compete with BLAS threads for CPU resources.

This explains the **very large performance gap** between serial and multiprocessing methods in the multithreaded BLAS results.

---

### 2. BLAS Forced to Single Thread

In the second experiment, BLAS was forced to run with a single thread using environment variables:

```
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

This removes internal BLAS parallelism.

Consequences:

* Serial computation becomes slower.
* Multiprocessing gains some relative advantage.

However, even in this configuration multiprocessing still performs worse for most tested problem sizes because:

* process management overhead remains significant
* memory duplication costs remain high

Only for very large (N) might multiprocessing begin to show benefits.

---

# Memory Usage Observations

Memory consumption increases significantly for both multiprocessing and block-wise approaches.

### Multiprocessing Memory Overhead

Each process may hold:

* partial datasets
* intermediate correlation results
* temporary buffers

This leads to higher overall memory usage compared to the serial implementation.

### Block-wise Memory Overhead

Block-wise computation introduces temporary matrices for each block multiplication, increasing peak memory usage during execution.

---

# Summary of Observed Behavior

| Method                  | Main Characteristics                           | Observed Performance                      |
| ----------------------- | ---------------------------------------------- | ----------------------------------------- |
| **Serial CPU**          | Uses optimized BLAS matrix multiplication      | Fastest                                   |
| **Multiprocessing CPU** | Python-level parallelism with process overhead | Slower due to overhead                    |
| **Block-wise CPU**      | Memory-efficient tiled computation             | Slightly slower due to smaller operations |

---

# Key Conclusion

The experiments demonstrate that **highly optimized numerical libraries can outperform naive parallelization strategies**.

Even though multiprocessing introduces parallel execution, its overhead and memory costs outweigh the benefits for moderate problem sizes.

In contrast, the serial implementation benefits from **optimized BLAS routines that already exploit hardware-level parallelism**, making it the most efficient approach for the tested dataset sizes.

The block-wise strategy, while slower here, remains valuable for **large-scale problems where memory constraints prevent full matrix computation**, and it aligns with strategies used in GPU and high-performance computing environments.

