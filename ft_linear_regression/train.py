import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: str) -> pd.DataFrame:

    try:
        data = pd.read_csv(path)
        return data
    except Exception as e:
        print(f"Error: {e}")


def model(X: np.ndarray, theta: np.ndarray):
    return X.dot(theta)


def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)


def grad(X, y, theta):
    m = len(y)
    # using ndarray's transpose `T()` function in the formula
    return 1/m * X.T.dot(model(X, theta) - y)


def gradient_descent(X, y, theta, learning_rate, n_iterations):

    cost_history = np.zeros(n_iterations)
    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(X, y, theta)
        cost_history[i] = cost_function(X, y, theta)
    return theta, cost_history


def train():

    df = load_csv("../data.csv")
    x, y = df["km"].to_numpy(), df["price"].to_numpy()

    x = x.reshape(x.shape[0], 1)
    y = y.reshape(y.shape[0], 1)

    # X matrix
    X = np.hstack((x, np.ones(x.shape)))
    theta = np.zeros((2, 1))
    learning_rate = 0.01
    n_iterations = 1000

    theta_final, cost_history = gradient_descent(X, y, theta,
                                                 learning_rate,
                                                 n_iterations)

    print(f"theta_final = {theta_final}")

    predictions = model(X, theta_final)
    plt.scatter(x, y)
    plt.plot(x, predictions, c='r')
    # plt.plot(range(1000), cost_history)
    plt.show()


def main():
    train()


if __name__ == "__main__":
    main()
