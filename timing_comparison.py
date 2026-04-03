import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import psutil

import yfinance as yf

def load_time_series():
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "TSLA", "NVDA", "JPM", "V", "UNH"
    ]

    data = yf.download(tickers, period="1y")["Close"]
    data = data.dropna()

    # Convert to returns
    returns = data.pct_change().dropna()

    return returns.values.T

# ---------------- GPU CORRELATION ----------------

def gpu_correlation_full(data: np.ndarray, device: torch.device):
    if device.type != "cuda":
        return None, 0, 0, 0, 0

    print("\n--- Running GPU Full Matrix ---")

    torch.cuda.reset_peak_memory_stats()

    # ---------------- TOTAL TIMER ----------------
    t_total_start = time.perf_counter()

    # ---------------- TRANSFER TIMER ----------------
    t_transfer_start = time.perf_counter()
    X = torch.tensor(data, device=device, dtype=torch.float64)
    torch.cuda.synchronize()
    t_transfer_end = time.perf_counter()

    # ---------------- COMPUTE TIMER ----------------
    t_compute_start = time.perf_counter()

    mean = X.mean(dim=1, keepdim=True)
    std  = X.std(dim=1, keepdim=True, unbiased=False)
    std  = torch.clamp(std, min=1e-8)

    Z = (X - mean) / std

    T = data.shape[1]
    corr_matrix = torch.mm(Z, Z.T) / T
    corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)

    torch.cuda.synchronize()
    t_compute_end = time.perf_counter()

    # ---------------- TOTAL END ----------------
    t_total_end = time.perf_counter()

    transfer_time = t_transfer_end - t_transfer_start
    compute_time  = t_compute_end - t_compute_start
    total_time    = t_total_end - t_total_start

    peak_mem_mb = torch.cuda.max_memory_allocated(0) / (1024**2)

    print(f"[GPU] Transfer Time: {transfer_time:.4f}s")
    print(f"[GPU] Compute Time:  {compute_time:.4f}s")
    print(f"[GPU] Total Time:    {total_time:.4f}s")
    print(f"[GPU] Peak VRAM:     {peak_mem_mb:.1f} MB")

    return corr_matrix, total_time, peak_mem_mb, transfer_time, compute_time

# ---------------- CPU CORRELATION ----------------
def cpu_corr(data):
    start = time.time()
    np.corrcoef(data)
    return time.time() - start

# ---------------- NUMERICAL CONSISTENCY ----------------
def check_numerical_consistency(data, gpu_corr):
    print("\n--- Numerical Consistency Check ---")

    # CPU result
    cpu_corr = np.corrcoef(data)

    # GPU → CPU → NumPy
    gpu_corr_np = gpu_corr.detach().cpu().numpy()

    # Absolute difference
    abs_diff = np.abs(cpu_corr - gpu_corr_np)

    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    print(f"Max abs difference:  {max_diff:.6e}")
    print(f"Mean abs difference: {mean_diff:.6e}")

    return max_diff, mean_diff

# ---------------- RAM ----------------
def get_ram_usage_mb():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)
    
# ---------------- MAIN EXPERIMENT ----------------
base_data = load_data()

factors = [1, 2, 5, 10, 20]

cpu_times = []
sizes = []
transfer_times = []
compute_times = []
ram_usage = []
sizes = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_faster_at = None

for f in factors:
    print(f"\n--- Scale factor: {f} ---")

    data = np.tile(base_data, (f, 1))
    print("Data shape:", data.shape)

    N = data.shape[0]
    sizes.append(N)

    # RAM before
    ram_before = get_ram_usage_mb()

    # CPU timing
    cpu_time = cpu_corr(data)
    cpu_times.append(cpu_time)

    # RAM after
    ram_after = get_ram_usage_mb()
    ram_used = ram_after - ram_before
    ram_usage.append(ram_used)

    # GPU timing 
    corr_matrix, gpu_time, gpu_mem, transfer_time, compute_time = gpu_correlation_full(data, device)
    gpu_times.append(gpu_time)
    transfer_times.append(transfer_time)
    compute_times.append(compute_time)

    print("\n--- Results ---")
    print(f"CPU time: {cpu_time:.4f}s")
    print(f"RAM used: {ram_used:.2f} MB")

    if corr_matrix is not None:
        print(f"GPU total: {gpu_time:.4f}s")
        print(f"   ↳ Transfer: {transfer_time:.4f}s")
        print(f"   ↳ Compute:  {compute_time:.4f}s")
        print(f"   ↳ VRAM:     {gpu_mem:.2f} MB")

        ratio = transfer_time / compute_time if compute_time > 0 else 0
        print(f"   ↳ Transfer/Compute ratio: {ratio:.2f}")

         # Bottleneck
        if transfer_time > compute_time:
            print("Bottleneck: GPU transfer")
        else:
            print("Bottleneck: GPU compute")
        
        # Crossover detection
        if gpu_faster_at is None and gpu_time < cpu_time: 
            gpu_faster_at = N
            
        check_numerical_consistency(data, corr_matrix)
        
# ---------------- CROSSOVER ----------------       
print("\n--- GPU vs CPU Crossover ---")
if gpu_faster_at:
    print(f"GPU becomes faster at N ≈ {gpu_faster_at}")
else: 
    print("GPU did not outperform CPU in tested range")

# ---------------- COMPLEXITY ANALYSIS PRINT ----------------
print("\n--- Complexity Analysis ---")
for i in range(1, len(sizes)):
    growth = cpu_times[i] / cpu_times[i - 1]
    print(f"N: {sizes[i-1]} → {sizes[i]} | Time growth: {growth:.2f}x")

print("\n--- GPU Complexity Analysis ---")
for i in range(1, len(sizes)):
    if gpu_times[i-1] > 0:  # avoid division issues
        growth = gpu_times[i] / gpu_times[i - 1]
        print(f"N: {sizes[i-1]} → {sizes[i]} | GPU growth: {growth:.2f}x")

# ---------------- PLOTTING ----------------

# CPU vs GPU
plt.figure()
plt.plot(sizes, cpu_times, marker='o', label='CPU')
plt.plot(sizes, gpu_times, marker='s', label='GPU')
plt.xlabel("N")
plt.ylabel("Time (s)")
plt.title("CPU vs GPU Scaling")
plt.legend()
plt.grid()
plt.show()

# RAM vs N
plt.figure()
plt.plot(sizes, ram_usage, marker='o')
plt.xlabel("N")
plt.ylabel("RAM Usage (MB)")
plt.title("RAM Usage vs N")
plt.grid()
plt.show()

# Transfer vs Compute
plt.figure()
plt.plot(sizes, transfer_times, marker='o', label='Transfer')
plt.plot(sizes, compute_times, marker='s', label='Compute')
plt.xlabel("N")
plt.ylabel("Time (s)")
plt.title("GPU Transfer vs Compute Time")
plt.legend()
plt.grid()
plt.show()
