# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from torch import Tensor
from typing_extensions import override


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer."""

    def __init__(self, sigma: float = 0.1):
        """Initialize the GaussianNoise module (gaussian dropout).

        Args:
            sigma (float): relative standard deviation used to generate the noise.

        Returns:
            GaussianNoise (nn.Module): Gaussian noise module.

        """
        super().__init__()
        self.sigma = sigma

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x * (1 + torch.randn_like(x) * self.sigma) if self.training and self.sigma > 0 else x
