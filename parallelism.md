# Parallelism in the CPU Experiment Pipeline

In this discussion we focus **only on how parallelism is used in the experiment pipeline**, ignoring the internal implementations of the helper functions.

---

## 1. What `run_cpu_experiments()` Actually Does

The function `run_cpu_experiments()` itself is **not parallel**.
It acts as an **experiment controller** that runs benchmarks for different dataset sizes.

The core loop is:

```python
for N in N_values:
    result = run_benchmark(N, T, num_workers)
```

Execution order:

```
N = 500   -> run benchmark
N = 1000  -> run benchmark
N = 2000  -> run benchmark
N = 4000  -> run benchmark
```

Each experiment runs one after another (**sequentially**).

This design is intentional because **benchmarks require clean and reliable timing measurements**.

---

## 2. Where the Parallelism Actually Happens

Parallelism occurs inside the benchmark step, where the **correlation matrix** is computed using multiple workers.

Conceptually, the correlation matrix is defined as:

$$
C_{ij} = \text{corr}(X_i, X_j)
$$

Where:

- $X_i$ = time series *i*  
- $X_j$ = time series *j*

The resulting matrix:

$$
C \in \mathbb{R}^{N \times N}
$$

Number of correlations that must be computed:

$$
\frac{N(N-1)}{2}
$$

This is the **computationally expensive part of the program**.


---

## 3. Why This Problem is Embarrassingly Parallel

In parallel computing, a problem is called **embarrassingly parallel** when it can be divided into many tasks that run completely independently, with almost **no communication or synchronization** required between them.

The term **“embarrassingly”** is used because the parallelism is so obvious and easy that it almost feels too simple.

For correlation matrix computation:

Example with **4 time series**:

```
(0,1)
(0,2)
(0,3)
(1,2)
(1,3)
(2,3)
```

Each pairwise correlation:

* Uses only (X_i) and (X_j)
* Does **not depend on other pairs**
* Can be **computed independently**

So workers can compute them **simultaneously**.

Example assignment:

```
Worker 1 → (0,1), (0,2)
Worker 2 → (0,3), (1,2)
Worker 3 → (1,3)
Worker 4 → (2,3)
```

Workers **do not need to communicate during computation**.

---

## 4. Typical CPU Parallel Strategy

The most common Python approaches are:

* `multiprocessing`
* `concurrent.futures`

Conceptual structure:

```
Main Process
     |
     |---- Worker 1 -> compute subset of correlations
     |---- Worker 2 -> compute subset of correlations
     |---- Worker 3 -> compute subset of correlations
     |---- Worker 4 -> compute subset of correlations
```

Each worker computes **different entries of the correlation matrix**.

---

## 5. Task Splitting Strategy

A common strategy is to **divide the correlation matrix by rows**.

Example worker assignment:

| Worker   | Rows handled  |
| -------- | ------------- |
| Worker 1 | row 0–999     |
| Worker 2 | row 1000–1999 |
| Worker 3 | row 2000–2999 |
| Worker 4 | row 3000–3999 |

Each worker computes:

```
corr(X_i, X_j) for j > i
```

Then fills both symmetric entries:

```
C[i,j] = value
C[j,i] = value
```

---

## 6. Why This Works Efficiently

Correlation computations are:

* **Independent**
* **Read-only on the dataset**
* **No shared intermediate writes**

Therefore:

* No synchronization overhead
* Minimal communication
* High parallel efficiency

---

## 7. Overall Architecture of the Experiment

The program effectively has **three layers**:

```
run_cpu_experiments()        (experiment driver)
        |
        | sequential loop over dataset sizes
        |
run_benchmark()              (measurement layer)
        |
        | calls serial and parallel implementations
        |
compute_correlation_parallel()
        |
        | splits work across workers
        |
Worker threads/processes
        |
        | compute correlations
```

---

## 8. Why Experiments Themselves Are Not Parallel

Technically, the experiments for different `N_values` **could run simultaneously**.

However, this would distort benchmarking results due to:

* CPU contention
* Memory contention
* Resource interference

Therefore, experiments are run **sequentially** to ensure **accurate timing measurements**.

---

## 9. Big Picture Parallel Model

```
Dataset X (N x T)
        |
        |
Divide correlation pairs
        |
        |
+-----------+-----------+-----------+-----------+
| Worker 1  | Worker 2  | Worker 3  | Worker 4  |
+-----------+-----------+-----------+-----------+
     |            |            |            |
 compute pairs compute pairs compute pairs compute pairs
     |            |            |            |
      -------- combine results --------
                   |
                   |
          Correlation Matrix C
```

---

## 10. Type of Parallelism Used

This project uses **data parallelism**.

The dataset is **partitioned across workers**, and each worker performs the **same computation on different subsets of the data**.

It is **not**:

* Task parallelism
* Pipeline parallelism

