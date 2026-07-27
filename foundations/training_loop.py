import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        
        n = X.shape[0]
        w = np.zeros(X.shape[1])
        b = 0.0
        for i in range(epochs): 
            # Forward pass   
            y_hat = X @ w + b
            erro = y_hat - y

            # MSE
            dw = (2.0/n) * (X.T @ erro)
            db = (2.0/n) * (np.sum(erro))

            #updates
            w = w - lr * dw
            b = b- lr * db


             


        return (np.round(w,5), np.round(b,5))
        pass
