import torch
import torch.nn as nn
import math
from typing import List

class Solution:
    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)

        weights = torch.empty(fan_out, fan_in)
        nn.init.xavier_normal_(weights)

        return torch.round(weights, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)

        weights = torch.empty(fan_out, fan_in)
        nn.init.kaiming_normal_(weights, mode="fan_in", nonlinearity="relu")

        return torch.round(weights, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)

        weights = []

        for layer in range(num_layers):
            fan_in = input_dim if layer == 0 else hidden_dim
            fan_out = hidden_dim

            weight_matrix = torch.empty(fan_out, fan_in)

            if init_type == "xavier":
                nn.init.xavier_normal_(weight_matrix)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(weight_matrix, mode="fan_in", nonlinearity="relu")
            else:
                weight_matrix = torch.randn(fan_out, fan_in)

            weights.append(weight_matrix)

        activations = torch.randn(input_dim)

        standard_deviations = []

        for weight_matrix in weights:
            activations = weight_matrix @ activations
            activations = torch.relu(activations)

            standard_deviations.append(round(activations.std().item(), 2))

        return standard_deviations