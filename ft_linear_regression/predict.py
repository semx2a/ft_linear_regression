import matplotlib.pyplot as plt
from train import Train
from time import sleep


def show_regression(model: Train):
    predictions = model.model(model.X, model.theta)
    plt.scatter(model.x, model.y, color='blue', label="Data")
    # plt.scatter(model.X[0:, 0], model.Y, color='green', label="")
    plt.plot(model.x, predictions, color='red', label="Regression line")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.legend()
    plt.title("Linear regression")
    plt.show()


def price_estimation(model: Train, mileage: int):
    price = model.theta[0] * mileage + model.theta[1]
    price = price[0]
    price = "{:.2f}".format(price)
    print(f"Price for {mileage} km: {price}", end="\n\n")


def show_cost_history(model: Train):
    plt.plot(range(model.n_iterations), model.cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost history")
    plt.show()


def main():
    model = Train("../data.csv")

    welcome = "Welcome to CarValue, your car price prediction machine!\n"
    welcome += "Please select an option:\n"
    welcome += "1. Estimate a price\n"
    welcome += "2. Show regression plot\n"
    welcome += "3. Show cost history\n"
    welcome += "4. Exit\n"
    welcome += "Your choice ▶ "
    predict = "What's the mileage (km) of the car to estimate the price of? "

    while True:
        try:
            choice = input(welcome)

            match choice:
                case choice if choice == "1":
                    data = input(predict)

                    if not data.isdigit():
                        print("\033[1;31m Please enter a valid number.\033[0m")
                        continue
                    mileage = int(data)
                    price_estimation(model, mileage)
                case choice if choice == "2":
                    show_regression(model)
                case choice if choice == "3":
                    show_cost_history(model)
                case choice if choice == "4":
                    print("\nExiting...")
                    break
                case _:
                    print("\033[1;31mPlease try again.\033[0m")
                    sleep(1)

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
