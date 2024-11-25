import matplotlib.pyplot as plt
from .train import Train


def hypothesis(theta: list, x: float | list) -> float | list:
    """ Linear regression hypothesis function.

    Parameters:
        theta (list): The model's parameters.
        x (float | list): The mileage of the car.
    Returns:
        float | list: The estimated price of the car.
    """
    return theta[0] * x + theta[1]


def show_regression(model: Train) -> None:
    """ Show the regression plot.

    Parameters:
        model (Train): The trained model.
    Returns:
        None
    """
    if model is None or model.payload is None:
        print("Please train the model first.", end="\n\n")
        return

    plt.scatter(model.x, model.y, color='blue', label="Data")

    predictions = hypothesis(model.payload["theta"], model.x)
    plt.plot(model.x, predictions, color='red', label="Regression line")

    plt.xlabel("Mileage")
    plt.ylabel("Price")

    plt.legend()
    plt.title("Linear regression")

    plt.show()


def show_cost_history(model: Train) -> None:
    """ Show the cost history plot.

    Parameters:
        model (Train): The trained model.
    Returns:
        None
    """
    if model is None or model.payload is None:
        print("Please train the model first.", end="\n\n")
        return

    plt.plot(range(model.payload["n_iterations"]),
             model.payload["cost_history"],
             color='blue',
             label="Cost history")

    plt.xlabel("Iterations")
    plt.ylabel("Cost")

    plt.legend()
    plt.title("Cost history")

    plt.show()


def price_estimation(model: Train, mileage: int) -> None:
    """ Estimate the price of a car given its mileage and prints it.

    Parameters:
        model (Train): The trained model.
        mileage (int): The mileage of the car.
    Returns:
        None
    """

    if model is None or model.payload is None:
        print("Please train the model first.", end="\n\n")
        return
    if mileage < 0 or mileage > 500000:
        print("Please enter a mileage between 0 and 500000.", end="\n\n")
        return

    price = model.payload["theta"][0] * mileage + model.payload["theta"][1]
    price = price[0]

    if price < 0:
        price = 0

    price = "{:.2f}".format(price)
    print(f"Price for {mileage}km: {price}", end="\n\n")


def coefficient_of_determination(model: Train) -> None:
    """ Calculate the coefficient of determination (R2 score) and prints it.

    Parameters:
        model (Train): The trained model.
    Returns:
        None
    """
    if model is None or model.payload is None:
        print("Please train the model first.", end="\n\n")
        return

    prediction = hypothesis(model.payload["theta"], model.x)
    y_bar = model.y.mean()

    ss_tot = ((model.y - y_bar) ** 2).sum()
    ss_res = ((model.y - prediction) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)

    print(f"R2 score: {r2}", end="\n\n")
