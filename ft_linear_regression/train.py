import numpy as np

from .load_csv import load
from .file_info import get_file_info
from .json import generate_json, load_json


class Train():
    """ Train a linear regression model. """

    def __init__(self, path: str):
        """ Initialize the training process.

        Parameters:
            path (str): The path to the dataset.
        """
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
            self.payload = self.__start_training()
            generate_json(name="carml", content=self.payload)

    def __start_training(self) -> dict:
        """ Start the training process.

        Parameters:
            None
        Returns:
            payload (dict): The trained model's payload.
        """
        self.X, self.Y = self.__normalize_data()
        self.__theta = np.zeros((2, 1))
        self.__learning_rate = 0.001
        self.__n_iterations = 5000

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

    def __get_features_targets(self) -> tuple:
        """ Get the features `x` and targets `y` from the dataset."""
        x = self.__df["km"].dropna().to_numpy()
        y = self.__df["price"].dropna().to_numpy()

        if len(x) != len(y):
            raise ValueError("Error: The dataset is not properly formatted.")

        return x, y

    def __normalize_data(self) -> tuple:
        """ Normalize the features and targets to prepare for training."""
        # transpose values into vertical stack
        vx = np.vstack(self.x)
        vy = np.vstack(self.y)

        # normalize values with robust scaler
        X = self.robust_scaler(vx)
        Y = self.robust_scaler(vy)

        # X matrix with 1s for bias
        X = np.hstack((X, np.ones(X.shape)))

        return X, Y

    def __build_payload(self) -> dict:
        """ Build the payload for the trained model.
        Payload contains theta, cost history, and number of iterations of the
        model."""
        return {
            "theta": self.__theta.tolist(),
            "cost_history": self.__cost_history.tolist(),
            "n_iterations": self.__n_iterations
        }

    @staticmethod
    def gradient_descent(X: np.ndarray,
                         Y: np.ndarray,
                         theta: np.ndarray,
                         learning_rate: float,
                         n_iterations: int) -> tuple:
        """ Perform gradient descent to train the model.

        Parameters:
            X (np.ndarray): The features.
            Y (np.ndarray): The targets.
            theta (np.ndarray): The model's parameters.
            learning_rate (float): The learning rate.
            n_iterations (int): The number of iterations.
        Returns:
            theta (np.ndarray): The trained model's parameters.
            cost_history (np.ndarray): The cost history.
        """
        cost_history = np.zeros(n_iterations)
        for i in range(0, n_iterations):
            theta = theta - learning_rate * Train.grad(X, Y, theta)
            cost_history[i] = Train.cost_function(X, Y, theta)
        return theta, cost_history

    @staticmethod
    def grad(X: np.ndarray, Y: np.ndarray,
             theta: np.ndarray) -> np.ndarray:
        """ Calculate the gradient of the cost function.

        Parameters:
            X (np.ndarray): The features.
            Y (np.ndarray): The targets.
            theta (np.ndarray): The model's parameters.
        Returns:
            np.ndarray: The gradient of the cost function.
        """
        m = len(Y)
        return 1/m * X.T.dot(Train.model(X, theta) - Y)

    @staticmethod
    def cost_function(X: np.ndarray, Y: np.ndarray,
                      theta: np.ndarray) -> float:
        """ Calculate the cost function.

        Parameters:
            X (np.ndarray): The features.
            Y (np.ndarray): The targets.
            theta (np.ndarray): The model's parameters.
        Returns:
            float: The cost of the model.
        """
        m = len(Y)
        return 1/(2*m) * np.sum((Train.model(X, theta) - Y)**2)

    @staticmethod
    def model(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """ Linear regression model.

        Parameters:
            X (np.ndarray): The features.
            theta (np.ndarray): The model's parameters.
        Returns:
            np.ndarray: The model's predictions.
        """
        return X.dot(theta)

    @staticmethod
    def robust_scaler(matrix: np.ndarray) -> np.ndarray:
        """ Normalize the matrix using the robust scaler.

        Parameters:
            matrix (np.ndarray): The matrix to normalize.
        Returns:
            np.ndarray: The normalized matrix.
        """
        q75, q25 = np.percentile(matrix, [75, 25])
        return (matrix - np.median(matrix)) / (q75 - q25)

    @staticmethod
    def denormalize_theta(theta: np.ndarray, x: np.ndarray,
                          y: np.ndarray) -> np.ndarray:
        """ Denormalize the theta values.
        Parameters:
            theta (np.ndarray): The model's parameters.
            x (np.ndarray): The features.
            y (np.ndarray): The targets.
        Returns:
            np.ndarray: The denormalized theta values.
        """
        theta[0] = theta[0] * y.std() / x.std()
        theta[1] = y.mean() - theta[0] * x.mean()
        return theta
