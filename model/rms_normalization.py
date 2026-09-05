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



# import torch
# from typing import List

# class Solution:

#     def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:

#         x = torch.tensor(x, dtype=torch.float32)
#         gamma = torch.tensor(gamma, dtype=torch.float32)

#         rms = torch.sqrt(torch.mean(x ** 2) + eps)

#         x_hat = x / rms

#         out = x_hat * gamma

#         return [round(v.item(), 4) for v in out]