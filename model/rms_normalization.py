import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        rms = np.sqrt(np.mean([xi * xi for xi in x]) + eps)
        x_hat = [xi / rms for xi in x]
        out = [x_hat[i] * gamma[i] for i in range(len(x))]
        return np.round(out, 4).tolist()
