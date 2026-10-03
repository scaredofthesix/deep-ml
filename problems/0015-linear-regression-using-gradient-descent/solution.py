import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    m, n = X.shape
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    w = np.zeros((n,1))  # Initialize weights to zeros

    for x in range(iterations):
        error = X @ w - y
        gradient = (1/ m)* X.T @ error
        w -= alpha * gradient
    # Your code here: implement gradient descent

    return w.flatten()