import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    X = np.array(X, dtype=float)
    weights = np.array(weights, dtype=float)

    z = X @ weights + bias
    p = sigmoid(z)

    return [1 if i >= 0.5 else 0 for i in p]