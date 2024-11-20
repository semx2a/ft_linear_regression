import matplotlib.pyplot as plt
from .train import Train


def hypothesis(theta, x):
    return theta[0] * x + theta[1]


def show_regression(model: Train):
    predictions = hypothesis(model.theta, model.x)

    plt.scatter(model.x, model.y, color='blue', label="Data")
    plt.scatter(model.X[0:, 0], model.Y, color='green',
                label="Normalized data")
    plt.plot(model.x, predictions, color='red', label="Regression line")

    plt.xlabel("Mileage")
    plt.ylabel("Price")

    plt.legend()
    plt.title("Linear regression")

    plt.show()


def price_estimation(model: Train, mileage: int):
    price = model.theta[0] * mileage + model.theta[1]
    price = price[0]

    if price < 0:
        price = 0

    price = "{:.2f}".format(price)
    print(f"Price for {mileage}km: {price}", end="\n\n")


def show_cost_history(model: Train):
    plt.plot(range(model.n_iterations), model.cost_history, 
             color='blue', label="Cost history")

    plt.xlabel("Iterations")
    plt.ylabel("Cost")

    plt.legend()
    plt.title("Cost history")

    plt.show()
