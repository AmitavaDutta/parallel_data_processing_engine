# What We Actually Did: Parallelization Strategy

We parallelized the **rows of the correlation matrix computation**, rather than splitting the raw data points into independent groups.

---

# The Mathematical Mapping

Recall the correlation formula used in our implementation:

\[
C = \frac{Z Z^T}{T-1}
\]

Where:

- \(Z\) (Standardized Matrix) shape = \((N, T)\)  
- \(C\) (Correlation Matrix) shape = \((N, N)\)  

Each row \(i\) of the resulting matrix \(C\) represents the correlation of time series \(i\) with all \(N\) time series in the dataset.

To parallelize this, we distribute the responsibility of calculating these rows across multiple CPU cores.

---

# Work Distribution (Example with 4 Workers)

We split the \(N\) rows of the target matrix \(C\) into chunks:

```
Worker 1 → computes rows 0 to 249 of C
Worker 2 → computes rows 250 to 499 of C
Worker 3 → computes rows 500 to 749 of C
Worker 4 → computes rows 750 to 999 of C
```

Each worker \(i\) extracts a subset of standardized rows \(Z_{chunk}\) and performs the multiplication against the entire transposed matrix:

\[
\text{Partial Result}_i = \frac{Z_{chunk} \times Z^T}{T-1}
\]

---

# The Shared Memory Architecture

In the optimized implementation, we replaced standard multiprocessing (which duplicates data) with a **Shared Memory Model**.

---

## 1. Zero-Copy Data Access

- A block of **system shared memory** is allocated.  
- The parent process writes the standardized matrix \(Z\) into this buffer once.  
- Each worker receives a reference (shared memory name) to access the same physical memory.  

This eliminates:

- Data duplication  
- Serialization (pickling) overhead  

---

## 2. Implementation Workflow

```
Main Process: Allocate Shared Memory → Write Z → Spawn Workers
      ↓
Worker Processes: Attach to Shared Memory → Compute C[start:end, :] → Return
      ↓
Main Process: Concatenate Results (vstack) → Unlink Shared Memory
```

---

# Why We Did It This Way

Correlation is an **all-to-all operation**.

Every time series must interact with every other time series to fill the \(N \times N\) matrix.

If we had split the dataset like:

```
Group A: Series 1–50
Group B: Series 51–100
```

Then cross-correlations such as:

```
Series 10 vs Series 90
```

would not be computed without additional communication.

By splitting **rows of the output matrix** while keeping the full dataset accessible via shared memory:

- All \(N^2\) correlations are computed  
- Redundant data movement is avoided  

---

# Why Performance Challenges Persist

Even after eliminating data transfer overhead using shared memory, the parallel CPU version often underperforms for moderate \(N\).

---

## 1. The "Process Tax"

- Process creation in Linux/Windows is expensive  
- Workers must initialize and attach to shared memory  

For small to moderate \(N\):

```
Process overhead > computation time
```

---

## 2. BLAS Native Supremacy

The serial implementation uses NumPy’s `dot`, backed by BLAS.

- Written in C/Fortran  
- Optimized for cache, SIMD, and threading  

Two problematic scenarios:

### BLAS Single-threaded

```bash
OPENBLAS_NUM_THREADS=1
```

- Loses internal parallelism  
- Reduces vectorized throughput  

### BLAS Multithreaded + Multiprocessing

- Each worker invokes multithreaded BLAS  
- Leads to **oversubscription**  

Example:

```
4 workers × 4 BLAS threads = 16 threads
```

Result:

- CPU contention  
- Context switching  
- Reduced performance  

---

## 3. Memory Hierarchy & Cache Locality

- Serial execution allows optimal cache reuse (L1/L2/L3)  
- Parallel workers access shared memory simultaneously  

This leads to:

- Cache contention  
- Increased memory bandwidth pressure  
- Reduced effective throughput  

---

# Summary of the Workflow

```
Prepare:   Standardize data and move it to Shared Memory
Execute:   Assign chunks of the N × N matrix to workers
Assemble:  Stack partial results into final matrix
Monitor:   Track memory usage across parent + worker processes
```

---

# Final Insight

Shared memory removes **data transfer overhead**, but does not eliminate:

- Process management cost  
- BLAS vs multiprocessing conflicts  
- Memory hierarchy limitations  

Performance is ultimately constrained by:

```
hardware efficiency > parallel abstraction
```

