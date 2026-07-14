import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:


        x = np.array(x)
        W1 = np.array(W1)
        W2 = np.array(W2)
        y_true = np.array(y_true)
        
        z1 = x @ W1.T + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2.T + b2

        loss = np.mean((z2 - y_true) ** 2)
        n = len(y_true) if len(y_true) > 0 else 1
        db2 = (2.0 * (z2 - y_true)) / n
        dW2 = np.outer(db2, a1)

        da1 = db2 @ W2
        relu_grad = (z1 > 0).astype(float)
        dz1 = da1 * relu_grad
        dW1 = np.outer(dz1, x)
            
        return {
            "loss" : np.round(loss, 4),
            "dW1" : np.round(dW1, 4),
            "db1" : np.round(dz1, 4),
            "dW2" : np.round(dW2, 4),
            "db2" : np.round(db2, 4)
        }




        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        pass
