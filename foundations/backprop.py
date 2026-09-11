import numpy as np
from numpy.typing import NDArray
from typing import Tuple
class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z = np.dot(x,w)+b
        y_pred = 1/(1+ np.exp(-z))
        error = y_pred-y_true
        sigmoid_gradient = y_pred*(1-y_pred)
        delta = error*sigmoid_gradient
        dl_dw = delta*x
        dl_db = delta
        return np.round(dl_dw, 5), round(float(dl_db), 5)