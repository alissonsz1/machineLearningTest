import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:

        x = np.array(x)
        beta = np.array(beta)
        gamma = np.array(gamma)
        running_mean = np.array(running_mean, dtype=np.float64)
        running_var = np.array(running_var, dtype=np.float64)


        if training:
            x_hat = (x - np.mean(x, axis=0))/(np.sqrt(np.var(x, axis=0) + eps))
            running_mean = np.round(((1.0 - momentum) * running_mean) + momentum * np.mean(x, axis=0),4)
            running_var = np.round((1.0 - momentum) * running_var + momentum * np.var(x, axis=0), 4)
            y = np.round(gamma * x_hat + beta,4)
            return (y, running_mean, running_var)


        else:
            x_hat = np.round((x - running_mean)/ (np.sqrt(running_var + eps)), 4)
            return (x_hat, running_mean, running_var)
          
          
          
        return (y, running_mean, running_var)
        
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        pass
