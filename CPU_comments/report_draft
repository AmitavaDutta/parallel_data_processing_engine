# Parallel Data Processing Engine: CPU Correlation Matrix Estimation

## 1. Project Objectives
The CPU component of this project establishes the high-performance baseline for evaluating large-scale correlation matrix estimation. The primary objectives were:
* **Serial Baseline:** Implement a reference version using vectorized NumPy operations for accuracy verification.
* **Multiprocessing Engine:** Design a parallel architecture using Python’s `shared_memory` to minimize data serialization overhead and bypass the Global Interpreter Lock (GIL).
* **Block-wise Computation:** Implement a tiling strategy to evaluate CPU cache locality and manage memory for datasets exceeding physical RAM.
* **System Profiling:** Quantify the interaction between user-defined process-level parallelism and internal library-level multithreading (BLAS).

## 2. Methodology & Algorithms
The engine computes the Pearson correlation matrix for $N$ time series of length $T$ using the following mathematical pipeline:

1.  **Standardization:** Time series are transformed into Z-scores: $Z = (X - \mu) / \sigma$.
2.  **Gramian Product:** The correlation matrix $C$ is derived via the product: $C = \frac{1}{T-1}(ZZ^T)$.
3.  **Symmetry Optimization:** In "Optimized" versions, we exploit the property $C_{ij} = C_{ji}$. By computing only the upper triangle, we theoretically reduce the computational workload by **50%**.
4.  **Hardware Control:** To isolate implementation performance, environment variables (`OPENBLAS_NUM_THREADS=1`) were used to toggle between single-core and multi-core BLAS configurations.

## 3. Empirical Results and Technical Analysis

### A. The Conflict of Parallelism (Multithreaded BLAS)
When the underlying BLAS library was permitted to use all available cores, manual parallelization (Multiprocessing) resulted in **negative speedup** (Speedup < 1.0).
* **Observation:** The Serial CPU version performed best because NumPy’s internal C-kernels are already parallelized. 
* **Insight:** Adding Python-level `multiprocessing` created **Resource Oversubscription**. The CPU spent more cycles on context-switching and Inter-Process Communication (IPC) than on actual computation.

### B. Scalability in Restricted Environments (Single-threaded BLAS)
By forcing BLAS to a single thread, the efficiency of the custom parallel architectures became evident:
* **Baseline Multiprocessing:** Showed strong scaling, achieving up to **3.5x speedup** on 4 cores, validating the shared-memory architecture.
* **The Optimization Paradox:** The "Optimized" multiprocessing version (using nested loops for symmetry) was significantly **slower** than the "Baseline" (full matrix multiplication). 
* **Insight:** This highlights the **Python Interpreter Overhead**. Even with 50% fewer math operations, executing millions of `np.dot` calls within Python loops is less efficient than a single, massive call to a highly-optimized C/Fortran kernel.

### C. Memory Trends
Memory usage scaled quadratically $O(N^2)$. At $N=20,000$, peak memory reached approximately **15 GB** for multiprocessing versions due to the overhead of managing shared buffers and process forks, whereas the Serial version required only **~4 GB**.

## 4. Current Status & Team Work Distribution
* **Status:** CPU implementations (Serial, Parallel, and Block-wise) are fully functional. Numerical consistency has been verified using `np.allclose()`.
* **Distribution:**
    * **Amitava:** Lead Developer for CPU-tier architecture. Implemented shared-memory multiprocessing, block-wise logic, and the benchmarking/visualization suite.
    * **Team Members:** Currently developing GPU-accelerated kernels (CuPy) and finalizing the cross-architecture report.

## 5. Expected Output
The CPU suite provides the "Performance Floor." For $N=20,000$, the engine completes the task in under **5 seconds** using multithreaded BLAS. The upcoming GPU implementation must demonstrate significant speedup to justify the overhead of Host-to-Device data transfers.# 1. Project Objectives (CPU Context)

The primary objective was to establish a **high-performance CPU baseline** for correlation matrix estimation. This involved:

- Implementing a **Serial Baseline** using vectorized NumPy operations  
- Designing a **Multiprocessing Engine** using shared memory to minimize data duplication  
- Developing a **Block-wise Computation strategy** to manage memory pressure and improve cache locality  
- Evaluating **BLAS multithreading vs. manual process-level parallelism**  

---

# 2. Algorithms & Mathematical Process

The engine computes the **Pearson correlation matrix** for \(N\) time series of length \(T\).

### Standardization

\[
Z = \frac{X - \mu}{\sigma}
\]

### Matrix Product

\[
C = \frac{1}{T-1}(Z Z^T)
\]

### Complexity

\[
O(N^2 \cdot T)
\]

For \(N = 20{,}000\):

- Total elements ≈ **400 million**
- Requires efficient **memory management + parallel execution**

---

## Symmetry Optimization

The correlation matrix satisfies:

\[
C_{ij} = C_{ji}
\]

In theory:

- Only upper triangle needs computation  
- Reduces operations by ~50%  

---

# 3. Technical Analysis & Current Status

---

## The "Oversubscription" Trap (Multithread BLAS)

### Observation

- Serial CPU (NumPy) was fastest  
- Parallel implementations showed **negative speedup (< 1.0)**  

### Reason

- NumPy’s `@` already uses all CPU cores via BLAS  
- Adding multiprocessing causes:

  - CPU oversubscription  
  - Context switching  
  - IPC overhead  

### Insight

```
Hardware-level parallelism > Python-level parallelism
```

---

## The Success of Manual Parallelism (Single-thread BLAS)

When BLAS is restricted:

```bash
OPENBLAS_NUM_THREADS=1
```

### Results

- Multiprocessing achieved **~3.5× speedup (4 cores)** for large \(N\)  
- Confirms shared-memory design effectively bypasses the GIL  

---

## The Optimization Paradox

### Observation

- "Optimized" version (upper triangle only) was **slower**  
- "Baseline" (full matrix multiplication) was **faster**  

### Reason

- Optimized version uses Python loops + many `np.dot` calls  
- Baseline uses **one large BLAS call**  

### Insight

```
Fewer operations ≠ faster execution
```

Performance depends on:

- C-level vectorization  
- Reduced Python interpreter overhead  

---

## Memory Scaling

- Memory scales as \(O(N^2)\)  

### Observations

| Implementation | Memory Usage (N = 20,000) |
|--------------|---------------------------|
| Serial CPU | ~4 GB |
| Multiprocessing | ~15 GB |

### Insight

- Multiprocessing introduces **memory overhead ("Memory Tax")**  
- Critical constraint for large geophysical datasets  

---

# 4. Team Work Distribution

### Student A (CPU Lead)

- Implemented:
  - Serial CPU  
  - Multiprocessing (Shared Memory)  
  - Block-wise computation  
- Benchmarked across:
  - BLAS Single-thread  
  - BLAS Multi-thread  
- Developed:
  - Visualization tools  
  - Speedup + memory analysis  

---

### Student B / C / D

- GPU implementation (CUDA / CuPy)  
- System integration  
- Final report synthesis  

---

# Expected Output (CPU Section)

The final CPU engine establishes a **performance baseline**:

- Handles \(N = 20{,}000\)  
- Runtime: **< 5 seconds** (Multithreaded BLAS)  

---

## Final Insight

This defines the:

```
Performance Floor
```

Any GPU implementation must:

- Exceed this runtime  
- Justify **CPU → GPU memory transfer overhead**
