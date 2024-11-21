import numpy as np

from .load_csv import load
from .file_info import get_file_info
from .json import generate_json, load_json


class Train():

    def __init__(self, path: str):

        self.__df = load(path)
        self.x, self.y = self.__get_features_targets()
        self.__file_info = load_json("file_info.json")
        self.payload = None

        db_info = get_file_info(path)
        if db_info != self.__file_info:
            self.__file_info = db_info
            generate_json(name="file_info", content=self.__file_info)

        data = load_json("carml.json")
        if data:
            if "theta" in data and "cost_history" in data:
                self.payload = {
                    "theta": np.array(data["theta"]),
                    "cost_history": np.array(data["cost_history"]),
                    "n_iterations": int(data["n_iterations"])
                    }
                print("Model already trained.")
        else:
            self.payload = self.__start_training(path)
            generate_json(name="carml", content=self.payload)

    def __start_training(self, path: str) -> dict:

        self.X, self.Y = self.__normalize_data()
        self.__theta = np.zeros((2, 1))
        self.__learning_rate = 0.001
        self.__n_iterations = 10000

        self.__theta, self.__cost_history = self.gradient_descent(
            self.X,
            self.Y,
            self.__theta,
            self.__learning_rate,
            self.__n_iterations
        )

        self.__theta = self.denormalize_theta(self.__theta, self.x, self.y)

        payload = self.__build_payload()
        return payload

    def __get_features_targets(self):
        """ Get the features and targets from the dataset.
        x = features,
        y = targets
        """
        x = self.__df["km"].dropna().to_numpy()
        y = self.__df["price"].dropna().to_numpy()

        if len(x) != len(y):
            raise ValueError("Error: The dataset is not properly formatted.")

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

    def __build_payload(self):
        return {
            "theta": self.__theta.tolist(),
            "cost_history": self.__cost_history.tolist(),
            "n_iterations": self.__n_iterations
        }

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
