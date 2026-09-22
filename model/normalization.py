import numpy as np

from numpy.typing import NDArray


class Solution:

    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:

        mean = np.mean(x)
        variance = np.mean((x - mean) ** 2)
        eps = 1e-5

        x_hat = (x - mean) / np.sqrt(variance + eps)

        y = gamma * x_hat + beta

        return np.round(y, 5)