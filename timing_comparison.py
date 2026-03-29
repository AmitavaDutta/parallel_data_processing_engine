import numpy as np
import time
import torch
import matplotlib.pyplot as plt

from cancer_data import load_data


# ---------------- GPU CORRELATION ----------------
def gpu_correlation_full(data: np.ndarray, device: torch.device):
    """Compute the entire N×N correlation matrix in a single GPU operation."""
    if device.type != "cuda": return None, 0, 0
    print("\n--- Running GPU Full Matrix ---")

    torch.cuda.reset_peak_memory_stats() # Reset memory tracker
    t0 = time.perf_counter()

    X = torch.tensor(data, device=device)
    mean = X.mean(dim=1, keepdim=True)
    std  = X.std(dim=1, keepdim=True, unbiased=False)
    std  = torch.clamp(std, min=1e-8)
    Z    = (X - mean) / std

    T = data.shape[1]
    # Matrix multiplication for correlation
    corr_matrix = torch.mm(Z, Z.T) / T
    corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)

    torch.cuda.synchronize() # Wait for GPU to finish
    t1 = time.perf_counter()

    exec_time = t1 - t0
    peak_mem_mb = torch.cuda.max_memory_allocated(0) / (1024**2)

    print(f"[GPU Full] Execution Time: {exec_time:.4f}s")
    print(f"[GPU Full] Peak VRAM: {peak_mem_mb:.1f} MB")

    return corr_matrix, exec_time, peak_mem_mb

# ---------------- CPU CORRELATION ----------------
def cpu_corr(data):
    start = time.time()
    np.corrcoef(data)
    return time.time() - start

# ---------------- MAIN EXPERIMENT ----------------
base_data = load_data()

factors = [1, 2, 5, 10, 20]

cpu_times = []
sizes = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for f in factors:
    print(f"\n--- Scale factor: {f} ---")

    data = np.tile(base_data, (f, 1))
    print("Data shape:", data.shape)

    N = data.shape[0]
    sizes.append(N)

    # CPU timing
    cpu_time = cpu_corr(data)
    cpu_times.append(cpu_time)

    # GPU timing (if available)
    _, gpu_time, gpu_mem = gpu_correlation_full(data, device)

    print("\n--- Results ---")
    print(f"CPU time: {cpu_time:.4f}s")

    if gpu_time != 0:
        print(f"GPU time: {gpu_time:.4f}s")
        print(f"GPU memory: {gpu_mem:.2f} MB")
    else:
        print("GPU not available")


# ---------------- COMPLEXITY ANALYSIS PRINT ----------------
print("\n--- Complexity Analysis ---")
for i in range(1, len(sizes)):
    growth = cpu_times[i] / cpu_times[i - 1]
    print(f"N: {sizes[i-1]} → {sizes[i]} | Time growth: {growth:.2f}x")

# ---------------- PLOTTING ----------------
plt.plot(sizes, cpu_times, marker='o')
plt.xlabel("Number of Time Series (N)")
plt.ylabel("CPU Time (seconds)")
plt.title("Scaling of Correlation Computation (O(N^2))")
plt.grid()
plt.show()
