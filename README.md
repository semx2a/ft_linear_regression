# ft_linear_regression

## The project

This school project is an introduction to machine learning. The goal is to predict the price of a car based on its mileage. The project is divided into two parts: a training part and a prediction part.

## What is linear regression?

Linear regression is the one of the founding principles of machine learning.

Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to "learn" (i.e., progressively improve performance on a specific task) from data, without being explicitly programmed.

## How does this project use linear regression?

This project aims at producing a model that can predict the price of a car based on its mileage.

The project is made of two parts: a training part and a prediction part.

### The training part

The training part is based on the following hypothesis:

```math
\begin{equation*}
error = \frac{1}{2m} \sum_{i=1}^{m} (estimatePrice(mileage_i) - price_i)^2
\end{equation*}
```

Where $m$ is the number of samples in the dataset, $mileage_i$ is the mileage of the $i-th$ car in the dataset, and $price_i$ is the price of the $i-th$ car in the dataset.

This hypothesis is a cost function that measures the error of the model. The goal of the training part is to find the values of the parameters $\theta_0$ and $\theta_1$ that minimize the error of the model.

The values of the parameters $\theta_0$ and $\theta_1$ are found using the gradient descent algorithm. This algorithm is an optimization algorithm that is used to minimize the error of the model. They will be stored in a file `theta.json` that will be used in the prediction part.

### The prediction part

The prediction part is based on the following hypothesis:

```math
\begin{equation*}
estimatePrice(mileage) = \theta_0 + (\theta_1 \times mileage)
\end{equation*}
```

This hypothesis is based on the assumption that the price of a car is a linear function of its mileage. The goal of the prediction program is to predict the price of the car based on its mileage.

### The algorithm

This algorithm is detailed in code inside the [jupyter notebook](./notebook.ipynb) inside this repo.

## Usage

### Dataset

The dataset used to train the model is located in the `data.csv` file. This file contains two columns: `mileage` and `price`.

### Training

To train the model, you need to run the following command:

```bash
python train.py
```

This will generate a file called `theta.csv` that contains the values of the parameters of the model.

#### A word about normalization

I chose to implement Robust Scaling as a normalization method. This method is robust to **outliers**[^1] and is based on the following formula:

```math
\begin{equation*}
X_{scaled} = \frac{X - X_{median}}{IQR}
\end{equation*}
```

Where $X$ is the feature to scale, $X_{median}$ is the median of the feature, and $IQR$ is the interquartile range of the feature.

### Prediction

To predict the price of a car based on its mileage, you need to run the following command:

```bash
python predict.py
```

You will be prompted to enter the mileage of the car you want to predict the price.

## Installation

To install the dependencies, you need to run the following command:

```bash
pip install -r requirements.txt
```

## References

### Articles

- [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)
- [Coefficient of Determination](https://en.wikipedia.org/wiki/Coefficient_of_determination)
- [Theta](https://en.wikipedia.org/wiki/Theta)
- [Feature Scaling](https://en.wikipedia.org/wiki/Feature_scaling)
- [Normlization(statistics)](https://en.wikipedia.org/wiki/Normalization_(statistics))
  - [Robust Scaler](https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.RobustScaler.html)
  - [Standard Scaler](https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.StandardScaler.html)

### Videos

- [La régression linéaire](https://www.youtube.com/watch?v=wg7-roETbbM&t=27s&ab_channel=MachineLearnia)
- [Regression linéraire Numpy](https://youtu.be/vG6tDQc86Rs?list=PLO_fdPEVlfKqUF5BPKjGSh7aV9aBshrpY)
- [Normalisation des données](https://youtu.be/OGWwzm304Xs?t=946)

[^1]: outliers are values that are significantly different from the rest of the data and can distort the model. Robust scaling is a normalization method that is robust to outliers. My main goal in implementing this normalization method was to prevent this issue from happening.
