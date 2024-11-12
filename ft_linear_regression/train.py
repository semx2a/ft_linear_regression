import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: str) -> pd.DataFrame:

    try:
        data = pd.read_csv(path)
        return data
    except Exception as e:
        print(f"Error: {e}")


def model(X, theta):
    return X.dot(theta)


def train():

    df = load_csv("../data.csv")
    x, y = df["km"].to_numpy(), df["price"].to_numpy()

    x = x.reshape(x.shape[0], 1)
    y = y.reshape(y.shape[0], 1)

    X = np.hstack(x, np.ones(x.shape))
    theta = np.random.randn(2, 1)


def main():
    train()


if __name__ == "__main__":
    main()
