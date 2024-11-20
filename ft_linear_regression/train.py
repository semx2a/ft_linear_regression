import pandas as pd
import numpy as np


class Train():

    def __init__(self, path: str):
        self.df = self.__load_csv(path)
        self.x, self.y = self.__load_data()
        self.X, self.Y = self.__normalize_data()

        self.theta = np.zeros((2, 1))
        self.learning_rate = 0.001
        self.n_iterations = 5000

        self.theta, self.cost_history = self.gradient_descent(
            self.X,
            self.Y,
            self.theta,
            self.learning_rate,
            self.n_iterations
        )

        self.theta = self.denormalize_theta(self.theta, self.x, self.y)

    def __load_csv(self, path: str) -> pd.DataFrame:

        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            print(f"Error: {e}")

    def __load_data(self):
        x = self.df["km"].dropna().to_numpy()
        y = self.df["price"].dropna().to_numpy()

        if len(x) != len(y):
            exit("Error: The dataset is not properly formatted.")

        return x, y

    def __normalize_data(self):
        # transpose values into vertical stack
        vx = np.vstack(self.x)
        vy = np.vstack(self.y)

        # normalize values with robust scaler
        X = self.robust_scaler(vx)
        Y = self.robust_scaler(vy)

        # X matrix with 1s for bias
        X = np.hstack((X, np.ones(X.shape)))

        return X, Y

    @staticmethod
    def gradient_descent(X, Y, theta, learning_rate, n_iterations):
        cost_history = np.zeros(n_iterations)
        for i in range(0, n_iterations):
            theta = theta - learning_rate * Train.grad(X, Y, theta)
            cost_history[i] = Train.cost_function(X, Y, theta)
        return theta, cost_history

    @staticmethod
    def grad(X, Y, theta):
        m = len(Y)
        return 1/m * X.T.dot(Train.model(X, theta) - Y)

    @staticmethod
    def cost_function(X, Y, theta):
        m = len(Y)
        return 1/(2*m) * np.sum((Train.model(X, theta) - Y)**2)

    @staticmethod
    def model(X: np.ndarray, theta: np.ndarray):
        return X.dot(theta)

    @staticmethod
    def robust_scaler(matrix):
        q75, q25 = np.percentile(matrix, [75, 25])
        return (matrix - np.median(matrix)) / (q75 - q25)

    @staticmethod
    def denormalize_theta(theta, x, y):
        theta[0] = theta[0] * y.std() / x.std()
        theta[1] = y.mean() - theta[0] * x.mean()
        return theta
