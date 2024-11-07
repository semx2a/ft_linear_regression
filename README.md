# ft_linear_regression

## The project

This school project is an introduction to machine learning. The goal is to predict the price of a car based on its mileage. The project is divided into two parts: a training part and a prediction part.

## The algorithm

1. Dataset: $(x, y)$ with $m$ examples, $n$ variables

```math
\begin{equation*}
\begin{matrix}
X = 
\begin{bmatrix} 
x^{(1)}_1 & x^{(1)}_2 & \cdots & x^{(1)}_n \\ 
x^{(2)}_1 & x^{(2)}_2 & \cdots & x^{(2)}_n \\ 
\vdots & \vdots & \ddots & \vdots \\ 
x^{(m)}_1 & x^{(m)}_2 & \cdots & x^{(m)}_n 
\end{bmatrix} \\

m \times (n+1)
\end{matrix}
\end{equation*}

\qquad

\begin{equation*}
\begin{matrix}
Y = 
\begin{bmatrix} 
y^{(1)} \\
y^{(2)} \\
\vdots \\
y^{(m)} 
\end{bmatrix} \\
m \times 1
\end{matrix}
\end{equation*}

\qquad

\begin{equation*}
\begin{matrix}
\theta = 
\begin{bmatrix} 
a \\ 
b 
\end{bmatrix} \\

(n + 1) \times 1 
\end{matrix}
\end{equation*}
```

2. Model:

```math
\begin{equation*}
F = X.\theta
\end{equation*}
\qquad
m \times 1
```

3. Cost function:

```math
\begin{equation*}
J(\theta) = \frac{1}{2m}\sum(X.\theta+Y)^2
\end{equation*}
\qquad
1 \times 1
```

4. Gradients:

```math
\begin{equation*}
\frac{\partial J(\theta)}{\partial \theta} = \frac{1}{m} X^T.(X.\theta-Y)
\end{equation*}
\qquad
(n+1) \times 1
```

5. Gradient Descent Algorithm:

```math
\begin{equation*}
\theta := \theta - \alpha\frac{\partial J(\theta)}{\partial \theta}
\end{equation*}
\qquad
(n+1) \times 1
```

## References

- [La régression linéaire 🇫🇷](https://www.youtube.com/watch?v=wg7-roETbbM&t=27s&ab_channel=MachineLearnia)