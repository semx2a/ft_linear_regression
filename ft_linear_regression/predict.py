import matplotlib.pyplot as plt
from .train import Train


def hypothesis(theta, x):
    return theta[0] * x + theta[1]


def show_regression(model: Train):

    plt.scatter(model.x, model.y, color='blue', label="Data")
#    plt.scatter(model.X[0:, 0], model.Y, color='green',
#                label="Normalized data")

    predictions = hypothesis(model.payload["theta"], model.x)
    plt.plot(model.x, predictions, color='red', label="Regression line")

    plt.xlabel("Mileage")
    plt.ylabel("Price")

    plt.legend()
    plt.title("Linear regression")

    plt.show()


def price_estimation(model: Train, mileage: int):
    price = model.payload["theta"][0] * mileage + model.payload["theta"][1]
    price = price[0]

    if price < 0:
        price = 0

    price = "{:.2f}".format(price)
    print(f"Price for {mileage}km: {price}", end="\n\n")


def show_cost_history(model: Train):
    plt.plot(range(model.payload["n_iterations"]),
             model.payload["cost_history"],
             color='blue',
             label="Cost history")

    plt.xlabel("Iterations")
    plt.ylabel("Cost")

    plt.legend()
    plt.title("Cost history")

    plt.show()


def coefficient_of_determination(model: Train):
    prediction = hypothesis(model.payload["theta"], model.x)
    y_bar = model.y.mean()
    ss_tot = ((model.y - y_bar) ** 2).sum()
    ss_res = ((model.y - prediction) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    print(f"R2 score: {r2}", end="\n\n")
