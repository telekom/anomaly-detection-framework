# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from torch import Tensor
from torch.nn.modules.loss import _Loss
from typing_extensions import override

from .utils import reduce

__all__ = ["MaskedMSELoss", "MaskedMAELoss", "MaskedSmoothL1Loss"]


class MaskedMSELoss(_Loss):
    """Compute the Mean Squared Error (MSE) only over elements where the target is not all zeros.

    This version masks out positions where all target dimensions are zero.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Initializes the loss.

        Args:
            reduction (str): the reduction method. Can be one of ['mean', 'sum', 'none'].

        """
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}. Must be one of ['mean', 'sum', 'none'].")
        super().__init__()
        self.reduction = reduction

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the Mean Squared Error.

        Args:
            input (Tensor): the input tensor
            target (Tensor): the target tensor

        """
        mask = target.ne(0).any(dim=-1, keepdim=True).expand_as(target)
        output = (input - target) ** 2

        return reduce(output, reduction=self.reduction, mask=mask)


class MaskedMAELoss(_Loss):
    """Compute the Mean Absolute Error (MAE) only over elements where the target is not all zeros.

    This version masks out positions where all target dimensions are zero.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Initializes the loss.

        Args:
            reduction (str): the reduction method. Can be one of ['mean', 'sum', 'none'].

        """
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}. Must be one of ['mean', 'sum', 'none'].")
        super().__init__()
        self.reduction = reduction

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the Mean Absolute Error.

        Args:
            input (Tensor): the input tensor
            target (Tensor): the target tensor

        """
        mask = target.ne(0).any(dim=-1, keepdim=True).expand_as(target)
        output = (input - target).abs()

        return reduce(output, reduction=self.reduction, mask=mask)


class MaskedSmoothL1Loss(_Loss):
    """Compute the Smoothed L1 Loss only over elements where the target is not all zeros.

    This version masks out positions where all target dimensions are zero.
    """

    def __init__(self, reduction: str = "mean", beta: float = 1.0) -> None:
        """Initializes the loss.

        Args:
            reduction (str): the reduction method. Can be one of ['mean', 'sum', 'none'].
            beta (float): the smoothing parameter

        """
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}. Must be one of ['mean', 'sum', 'none'].")
        super().__init__()
        self.reduction = reduction
        self.beta = beta

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the Mean Absolute Error.

        Args:
            input (Tensor): the input tensor
            target (Tensor): the target tensor

        """
        mask = target.ne(0).any(dim=-1, keepdim=True).expand_as(target)

        output = (input - target).abs()
        output = torch.where(output < self.beta, 0.5 * output**2 / self.beta, output - 0.5 * self.beta)
        return reduce(output, reduction=self.reduction, mask=mask)
