# Why Parallel is Slower Most of the Time

The implementation uses:

`ThreadPoolExecutor`

and inside each thread you run:

```
chunk @ Z.T
```

However, NumPy matrix multiplication is already parallelized internally using **BLAS libraries** such as **OpenBLAS** or **Intel MKL**.

This means the **serial implementation already uses multiple CPU cores**.

When Python threads are added on top of this, you introduce additional overhead instead of gaining performance.

The added overhead includes:

1. Python thread management overhead  
2. Splitting matrices into chunks  
3. Thread scheduling by the OS  
4. Merging results using `vstack`  

Because of these costs, the parallel implementation often becomes **slower than the serial version**.

---

# 2. Why `N = 4000` Briefly Shows Speedup

```
speedup ≈ 1.06
```

This happens because:

- The computation becomes heavier  
- The overhead becomes relatively smaller  

However, this improvement is **not stable**, which is why performance drops again at **N = 8000** and **N = 10000**.

---

# 3. Why Parallel Memory Explodes at Large N

Look at the results:

| N | Serial | Parallel |
|---|---|---|
| 8000 | 1098 MB | 1814 MB |
| 10000 | 1454 MB | 2464 MB |

Parallel requires extra memory because:

- `chunks` copy parts of the matrix  
- Each thread holds intermediate products  
- Stacking results creates additional buffers  

Therefore, memory usage increases significantly.

This is **exactly the type of scaling behavior the project aims to demonstrate**.

---

# 4. Computational Complexity Explanation

The correlation matrix size is:

```
C = N × N
```

Memory complexity:

```
O(N²)
```

### Examples

| N | Correlation Matrix Size |
|---|---|
| 4000 | 16 million entries |
| 8000 | 64 million entries |
| 10000 | 100 million entries |

If each value is stored as an 8-byte float:

```
100M × 8 bytes ≈ 800 MB
```

This matches the **observed memory growth** in the measurements.

---

# 5. Why Speedup Drops Again at Large N

At larger matrix sizes:

- **Memory bandwidth becomes the bottleneck**
- **Thread contention increases**
- **NumPy's internal BLAS threading competes with Python threads**

This phenomenon is called:

**Oversubscription of CPU threads**

---

# 6. Why These Results Are Actually Good

The experimental results clearly demonstrate:

- ✔ Parallel overhead  
- ✔ Memory scaling behavior  
- ✔ Limited speedup  
- ✔ Correctness validation  
- ✔ `O(N²)` computational scaling  

These are **exactly the insights expected in a parallel computing project**.

---

# 7. Example Analysis for The Report

You could summarize the findings as follows:

> The multi-threaded CPU implementation did not consistently outperform the serial implementation. This occurs because NumPy’s underlying BLAS libraries already exploit multi-core parallelism during matrix multiplication. Adding Python-level threading introduces additional overhead such as thread management, memory partitioning, and result merging.  
>
> For moderate problem sizes (N ≈ 4000), the parallel version briefly approaches or slightly exceeds serial performance. However, as the matrix size increases, memory overhead and thread contention dominate, reducing the overall performance benefit.
