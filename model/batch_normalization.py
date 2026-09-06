import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        m=momentum
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)
        beta = np.array(beta, dtype=float)
        running_mean=np.array(running_mean)
        running_var=np.array(running_var)

        if training :
            b_mean= np.mean(x,axis=0)
            b_var=np.mean((x-b_mean)**2,axis=0)

            b_hat = (x - b_mean) / np.sqrt(b_var + eps)
            y=gamma*b_hat+beta

            running_mean=(1-m)*running_mean + m*b_mean
            running_var=(1-m)*running_var + m*b_var

        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)

            # Scale and shift
            y = gamma * x_hat + beta
        
        return (np.round(y, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist())




