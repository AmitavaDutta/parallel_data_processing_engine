# What We Actually Did

We parallelized **rows of the correlation matrix computation**, not raw data points.

Recall the correlation formula:

\[
C = \frac{Z Z^T}{T-1}
\]

Where:

- \(Z\) shape = \((N, T)\)
- \(C\) shape = \((N, N)\)

Each row of \(C\) is the correlation of **one time series with all others**.

Therefore, we split **rows of \(C\)** across processes.

Example with **4 workers**:

```
Worker 1 → rows 0–249
Worker 2 → rows 250–499
Worker 3 → rows 500–749
Worker 4 → rows 750–999
```

Each worker computes:

\[
Z_i @ Z^T
\]

for its assigned rows.

Then we stack the partial results to form the full matrix.

Conceptually:

```
Process 1 → compute C[0:250, :]
Process 2 → compute C[250:500, :]
Process 3 → compute C[500:750, :]
Process 4 → compute C[750:1000, :]
```

Final result:

```
C = concatenate(results)
```

So the workflow becomes:

```
parallel compute → partial results → merge
```

However, the **split dimension is the correlation matrix rows**, not the dataset itself.

---

# Why We Did It This Way

Correlation requires **all time series interacting with all others**.

If the dataset were split like this:

```
first 50 series → process 1
last 50 series → process 2
```

then **cross correlations would be missing**, such as:

```
series 1 vs series 60
series 2 vs series 80
```

These pairs would never be computed.

Therefore the correct strategy is to split by:

```
rows of the correlation matrix
```

not by

```
subsets of the dataset
```

---

# Computational Structure

The work distribution becomes:

Input:

```
Z (N × T)
```

Worker \(i\) computes:

\[
Z_i @ Z^T
\]

Where:

```
Zi = subset of rows of Z
```

Result shape:

```
Zi @ Z.T → (rows_i × N)
```

Then the final matrix is assembled:

```
C = vstack(results)
```

---

# Why This Still Performed Poorly

Even though the work is parallelized, each process still needs access to:

```
Z.T (size N × T)
```

This causes several problems:

- **Large memory duplication**
- **Heavy inter-process communication overhead**
- **BLAS already performing parallel computation internally**

Because of these factors, the multiprocessing implementation performs worse than the **NumPy serial implementation**.

