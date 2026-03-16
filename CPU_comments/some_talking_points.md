# CPU Implementations and Performance Analysis

---

## Slide 4: Transition to CPU Implementations

**Purpose:** Introduce the CPU phase of the project: design and evaluation.  

**Key Points:**

- Three CPU strategies benchmarked:  
  1. Serial CPU (NumPy)  
  2. Multiprocessing CPU  
  3. Block-wise CPU  

- Tested under two hardware configurations:  
  - BLAS Multithreaded (default NumPy behavior)  
  - BLAS Single-threaded (forcing sequential execution)  

- Objective: Isolate effects of **Python-level parallelism** versus **low-level library optimizations**.

---

## Slide 5: CPU Memory Usage Comparison (N up to 10,000)

**Purpose:** Analyze memory scaling with number of time series \(N\).  

**Talking Points:**

- **Serial CPU:** Lowest memory footprint; no overhead from multiple processes or temporary tiles.  
- **Multiprocessing CPU:** Higher memory due to duplication; each process maintains its own workspace.  
- **Block-wise CPU:** Intended for memory efficiency at massive scales, but higher peak memory here due to repeated temporary matrix allocations for tiles.

---

## Slide 6: CPU Runtime Comparison (N up to 10,000)

**Purpose:** Compare execution efficiency.  

**Talking Points:**

- **Serial CPU:** Fastest; NumPy delegates \(C = \frac{ZZ^T}{T-1}\) to highly optimized BLAS libraries (MKL/OpenBLAS).  
- **Multiprocessing CPU:** Slower; process creation and IPC (copying large arrays) dominate runtime for moderate \(N\).  
- **Block-wise CPU:** Slower due to loop overhead and smaller matrix operations, which cannot fully exploit BLAS optimizations.

---

## Slide 7: CPU Parallel Speedup

**Purpose:** Visualize "Speedup vs Serial".  

**Talking Points:**

- Speedup < 1.0 → parallel version slower than serial baseline.  
- For \(N \leq 10,000\), Python multiprocessing cannot overcome its own overhead.  
- Row-level parallelization of the correlation matrix is less efficient than hardware-level parallelism in the serial implementation (SIMD + cache-optimized routines).

---

## Slide 8: Scaling with Increased \(N\) Values

**Purpose:** Transition to larger datasets (\(N \leq 20,000\)) to observe scaling limits.  

**Talking Points:**

- Correlation matrix computation complexity grows quadratically: \(O(N^2)\).  
- Larger \(N\) amplifies performance gaps and resource constraints seen in earlier slides.

---

## Slide 9: Memory Usage Comparison (N up to 20,000)

**Purpose:** Examine memory consumption at larger scale.  

**Talking Points:**

- Memory curves steepen due to quadratic growth of \(C\) (100×100 MB → 400×400 MB equivalents).  
- Block-wise strategy becomes necessary when full matrix exceeds RAM, despite slower runtime in this environment.

---

## Slide 10: CPU Runtime Comparison (N up to 20,000)

**Purpose:** Final CPU performance benchmark.  

**Talking Points:**

- Serial implementation maintains the lead, proving highly optimized numerical libraries outperform naive parallelization.  
- Key takeaway: parallelism must be implemented at the **appropriate level** (hardware vs. application) to be effective.

---

## Slide 11: Transition to GPU Acceleration

**Purpose:** Introduce GPU-based acceleration phase.  

**Talking Points:**

- CPU limits arise from raw speed, memory bandwidth, and Python process overhead.  
- Next phase leverages **CuPy** to exploit GPU cores, ideal for massive \(O(N^2)\) matrix multiplications.

---

## Memory Usage and BLAS Performance Clarification

**Why memory usage is identical across BLAS modes:**

- Memory footprint depends on **data**, not number of threads.  
- Both modes store \(Z\) (\(N \times T\)) and \(C\) (\(N \times N\)) in RAM.  
- BLAS threads **share the same memory space**, unlike multiprocessing processes which duplicate data.  

**Factors contributing to slower speed despite identical memory:**

1. **Hardware Parallelism (SIMD + Multi-core)**  
   - Single-threaded BLAS uses 1 core; multithreaded uses all cores.  
   - SIMD vectorization throughput is limited in single-threaded mode.

2. **Python-Level Overheads (Multiprocessing)**  
   - IPC: Pickling/unpickling large matrices adds expensive computation.  
   - Process management: Spawning, synchronizing, and destroying processes exceeds computation savings for moderate \(N\).  
   - GIL is bypassed, but overhead remains significant.

3. **Cache Optimization**  
   - Serial BLAS implementations manage L1/L2/L3 caches efficiently.  
   - Custom parallel implementations often incur more cache misses, reducing effective throughput.

**Takeaway:** Memory footprint is not the limiting factor; **execution speed is dominated by hardware utilization and library-level optimizations**.

---

## Transition to GPU Strategy Comparison

- Next, compare **Full Matrix vs Block-wise GPU computation**, highlighting how GPUs handle massive \(O(N^2)\) correlations more efficiently.

