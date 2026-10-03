import numpy as np

def gradient_descent(X, y, weights,learning_rate,n_epochs,batch_size=1,method='batch'):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    w = np.array(weights, dtype=float)

    n = len(y)

    for _ in range(n_epochs):

        if method == "batch":
            error = X @ w - y
            gradient = (2 / n) * X.T @ error
            w -= learning_rate * gradient

        elif method == "stochastic":
            for i in range(n):
                xi = X[i]
                yi = y[i]

                error = xi @ w - yi
                gradient = 2 * xi * error

                w -= learning_rate * gradient

        elif method == "mini_batch":
            for start in range(0, n, batch_size):
                X_batch = X[start:start + batch_size]
                y_batch = y[start:start + batch_size]

                error = X_batch @ w - y_batch

                b = len(y_batch)
                gradient = (2 / b) * X_batch.T @ error

                w -= learning_rate * gradient

    return w