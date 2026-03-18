import numpy as np
import matplotlib.pyplot as plt
from dataset import generate_dataset, read_dataset

def main():

    N = 10
    T = 100

    #data = generate_dataset(N, T)
    data = read_dataset("../global_temperature_comparison_modified.csv")
    print("Dataset shape (N,T):", data.shape)

    # Compute correlation matrix
    corr_matrix = np.corrcoef(data)

    print("Correlation matrix shape (N,N):", corr_matrix.shape)

    # visualize correlation matrix
    plt.imshow(corr_matrix, cmap="coolwarm")
    plt.colorbar(label="Correlation")
    plt.title("Correlation Matrix")
    plt.xlabel("Series Index")
    plt.ylabel("Series Index")
    plt.show()

if __name__ == "__main__":
    main()
