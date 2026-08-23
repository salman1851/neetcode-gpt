import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        
        N = X.shape[0]
        d = X.shape[1]
        w = np.zeros(d)
        b = 0.0
        for _ in range(epochs):        
            y_hat = X @ w + b
            error = y_hat - y
            
            grad_w = (2.0/N) * (X.T @ error) 
            grad_b = (2.0/N) * np.sum(error)

            w = w - lr * grad_w
            b = b - lr * grad_b
        return np.round(w,5), np.round(b,5)
            