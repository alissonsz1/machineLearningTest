import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x = np.array(x);
        gamma = np.array(gamma)
        

        rms = np.sqrt(eps + np.mean(x**2));

        y = x/rms * gamma;

        return np.round(y,4)


        
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        pass
