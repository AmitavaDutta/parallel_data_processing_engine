import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


# ---------------- LOAD REAL TIME SERIES ----------------
def load_time_series():
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "TSLA", "NVDA", "JPM", "V", "UNH"
    ]

    # Download 1 year of daily closing prices
    data = yf.download(tickers, period="1y")["Close"]

    # Drop missing values
    data = data.dropna()

    # Convert to returns (IMPORTANT)
    returns = data.pct_change().dropna()

    # Return as numpy array (rows = time series)
    return returns.values.T, returns.columns


# ---------------- MAIN ----------------
base_data, labels = load_time_series()

print("Shape:", base_data.shape)
print("\nFirst 5 series (first 10 values):\n")
print(base_data[:5, :10])


# ---------------- PLOT TIME SERIES ----------------
plt.figure(figsize=(10, 6))

for i in range(min(5, base_data.shape[0])):
    plt.plot(base_data[i], label=labels[i])

plt.title("Sample Time Series (Stock Returns)")
plt.xlabel("Time (days)")
plt.ylabel("Returns")
plt.legend()
plt.grid()
plt.show()


# ---------------- DISTRIBUTION ----------------
plt.figure(figsize=(8, 5))
plt.hist(base_data.flatten(), bins=50)
plt.title("Distribution of Returns")
plt.xlabel("Return value")
plt.ylabel("Frequency")
plt.grid()
plt.show()


# ---------------- CORRELATION HEATMAP ----------------
corr = np.corrcoef(base_data)

plt.figure(figsize=(6, 6))
plt.imshow(corr, vmin=-1, vmax=1)
plt.colorbar()
plt.title("Correlation Matrix (Stocks)")
plt.xticks(range(len(labels)), labels, rotation=90)
plt.yticks(range(len(labels)), labels)
plt.show()
