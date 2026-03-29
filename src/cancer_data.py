from sklearn.datasets import load_breast_cancer
import pandas as pd

def load_data():
    data = load_breast_cancer()
    return data.data

if __name__ == "__main__":
    X = load_data()
    
    # show shape
    print("Shape:", X.shape)
    
    # show first few rows
    df = pd.DataFrame(X)
    print(df.head())
