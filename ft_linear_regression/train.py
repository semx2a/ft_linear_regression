import pandas as pd
import numpy as np


class Train():

    def __init__(self, path: str):
        self.df = self.load_csv(path)
        self.x, self.y = self.load_data()
        self.X, self.Y = self.normalize_data()
        self.theta = np.zeros((2, 1))
        self.learning_rate = 0.001
        self.n_iterations = 1000

        self.theta, self.cost_history = self.gradient_descent(
            self.X,
            self.Y,
            self.theta,
            self.learning_rate,
            self.n_iterations
            )

        self.theta = self.denormalize_theta(self.theta)
        print(f"theta = {self.theta}")

    def load_csv(self, path: str) -> pd.DataFrame:

        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            print(f"Error: {e}")

    def load_data(self):
        x = self.df["km"].dropna().to_numpy()
        y = self.df["price"].dropna().to_numpy()

        print(f"x: {x}")
        print(f"y: {y}")
        return x, y

    def normalize_data(self):
        # transpose values into vertical stack
        vx = np.vstack(self.x)
        vy = np.vstack(self.y)
        print(f"xv: {self.x}")
        print(f"yv: {self.y}")

        # normalize values
        X = self.robust_scaler(vx)
        print(f"X scaled: {X}")
        Y = self.robust_scaler(vy)
        print(f"Y scaled: {Y}")

        # X matrix
        X = np.hstack((X, np.ones(X.shape)))
        print(f"X normalized = {X}")

        return X, Y

    def robust_scaler(self, matrix):
        return (matrix - matrix.mean()) / matrix.std()

    def gradient_descent(self, X, Y, theta, learning_rate, n_iterations):
        cost_history = np.zeros(n_iterations)
        for i in range(0, n_iterations):
            theta = theta - learning_rate * self.grad(X, Y, theta)
            cost_history[i] = self.cost_function(X, Y, theta)
        return theta, cost_history

    def grad(self, X, Y, theta):
        m = len(Y)
        return 1/m * X.T.dot(self.model(X, theta) - Y)

    def cost_function(self, X, Y, theta):
        m = len(Y)
        return 1/(2*m) * np.sum((self.model(X, theta) - Y)**2)

    def model(self, X: np.ndarray, theta: np.ndarray):
        return X.dot(theta)

    def denormalize_theta(self, theta):
        return theta * self.y.std() / self.x.std()
