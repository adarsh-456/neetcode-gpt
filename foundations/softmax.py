import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z_shift=z-np.max(z)
        z_exp=np.exp(z_shift)
        return np.round(z_exp/np.sum(z_exp),4)
