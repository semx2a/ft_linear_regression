import matplotlib.pyplot as plt
from train import Train

model = Train("../data.csv")

predictions = model.model(model.X, model.theta)
plt.scatter(model.x, model.y, color='blue', label="Données réelles")
plt.scatter(model.X[0:, 0], model.Y, color='green', label="Données normalisées")
plt.plot(model.x, predictions, color='red', label="Ligne de regression")
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.legend()
plt.title("Régression linéaire avec coefficient de détermination")
# plt.plot(range(1000), cost_history)
plt.show()
