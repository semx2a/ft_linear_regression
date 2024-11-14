import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing    


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

    try:
        df = load_csv("../data.csv")
        x, y = df["km"].to_numpy(), df["price"].to_numpy()
        print(f"x: {x}")
        print(f"y: {y}")

        x = np.vstack(x)
        y = np.vstack(y)
        print(f"xv: {x}")
        print(f"yv: {y}")

        # normalize values
        X_robust = preprocessing.RobustScaler().fit_transform(x)
        print(f"X scaled: {X_robust}")
        Y_robust = preprocessing.RobustScaler().fit_transform(y)
        print(f"Y scaled: {Y_robust}")

        # X matrix
        X = np.hstack((x, np.ones(x.shape)))
        X_normalized = np.hstack((X_robust, np.ones(X_robust.shape)))
        print(f"X_normalized = {X_normalized}")
        theta = np.zeros((2, 1))
        learning_rate = 0.001
        n_iterations = 1000

        theta_final, cost_history = gradient_descent(X_normalized,
                                                     Y_robust,
                                                     theta,
                                                     learning_rate,
                                                     n_iterations)

        print(f"theta_final = {theta_final}")

        predictions = model(X, theta_final)
        plt.scatter(x, y, color='blue', label="Données réelles")
        # plt.scatter(X_robust, Y_robust)
        plt.plot(x, predictions, color='red', label="Ligne de regression")
        plt.xlabel("Mileage")
        plt.ylabel("Price")
        plt.legend()
        plt.title("Régression linéaire avec coefficient de détermination")
        # plt.plot(range(1000), cost_history)
        plt.show()

    except Exception as e:
        exit(f"Exception: {e}")


def main():
    train()


if __name__ == "__main__":
    main()
