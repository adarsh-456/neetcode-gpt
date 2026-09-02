import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)

        std = math.sqrt(2 / (fan_in + fan_out))

        weights = torch.randn(fan_out, fan_in) * std

        return torch.round(weights, decimals=4).tolist()


    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)

        std = math.sqrt(2 / fan_in)

        weights = torch.randn(fan_out, fan_in) * std

        return torch.round(weights, decimals=4).tolist()


    def check_activations(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        init_type: str
    ) -> List[float]:

        torch.manual_seed(0)

        # 1. Create all weight matrices first
        weights = []

        for layer in range(num_layers):

            fan_in = input_dim if layer == 0 else hidden_dim
            fan_out = hidden_dim

            if init_type == "xavier":
                std = math.sqrt(2 / (fan_in + fan_out))

            elif init_type == "kaiming":
                std = math.sqrt(2 / fan_in)

            elif init_type == "random":
                std = 1.0

            else:
                raise ValueError("Invalid init_type")

            W = torch.randn(fan_out, fan_in) * std

            weights.append(W)

        # 2. Generate random input AFTER creating all weights
        x = torch.randn(input_dim)

        # 3. Forward pass
        activation_stds = []

        for W in weights:

            x = x @ W.T

            x = torch.relu(x)

            activation_stds.append(round(x.std().item(), 2))

        return activation_stds