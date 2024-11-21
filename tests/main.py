import sys
from time import sleep

from ft_linear_regression import predict
from ft_linear_regression import train


def main(argv):

    if len(argv) != 2:
        print("\033[1;31mPlease provide a valid path to the dataset.\033[0m")
        sys.exit(1)

    model = train.Train(argv[1])

    welcome = "Welcome to CarML, your car price prediction machine!\n"
    welcome += "Please select an option:\n"
    welcome += "1. Estimate a price\n"
    welcome += "2. Show regression plot\n"
    welcome += "3. Show cost history\n"
    welcome += "4. Show R2\n"
    welcome += "5. Exit\n"
    welcome += "Your choice ▶ "
    irma = "What's the mileage (km) of the car to estimate the price of? "

    while True:
        try:
            choice = input(welcome)

            match choice:
                case choice if choice == "1":
                    data = input(irma)

                    if not data.isdigit():
                        print("\033[1;31m Please enter a valid number.\033[0m")
                        continue
                    mileage = int(data)
                    predict.price_estimation(model, mileage)
                case choice if choice == "2":
                    predict.show_regression(model)
                case choice if choice == "3":
                    predict.show_cost_history(model)
                case choice if choice == "4":
                    predict.coefficient_of_determination(model)
                case choice if choice == "5":
                    print("\nExiting...")
                    break
                case _:
                    print("\033[1;31mPlease try again.\033[0m")
                    sleep(1)

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main(sys.argv)
