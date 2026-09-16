# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from torch import Tensor
from torchmetrics.aggregation import BaseAggregator
from typing import Any
from typing_extensions import override


class MeanQuantileMetric(BaseAggregator):
    """Compute the quantile value of a given tensor."""

    def __init__(self, quantile: float | Tensor = 0.9999, nan_strategy: str | float = "warn", **kwargs: Any):
        """Initialize the metric with the specified interpolation method and back portion.

        Args:
            quantile (float | Tensor): quantile value to compute.
            nan_strategy: options:
                - ``'error'``: if any `nan` values are encountered will give a RuntimeError
                - ``'warn'``: if any `nan` values are encountered will give a warning and continue
                - ``'ignore'``: all `nan` values are silently removed
                - a float: if a float is provided will impute any `nan` values with this value

            kwargs: Additional keyword arguments, see :ref:`BaseAggregator kwargs` for more info.

        """
        super().__init__(
            "sum",
            torch.tensor(0.0, dtype=torch.get_default_dtype()),
            nan_strategy,
            state_name="mean_quantile_value",
            **kwargs,
        )
        self.quantile = quantile
        self.add_state("len", default=torch.tensor(0.0, dtype=torch.get_default_dtype()), dist_reduce_fx="sum")

    @override
    def update(self, value: float | Tensor) -> None:
        """Update the metric with a new value.

        Args:
            value (float | Tensor): The new value to update the metric with.
                If the value is not a Tensor, it will be converted to a Tensor with
                the specified dtype and device.

        Returns:
            None

        """
        # broadcast weight to value shape
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        self.mean_quantile_value += torch.quantile(value, self.quantile)
        self.len += 1

    @override
    def compute(self) -> Any | Tensor:
        """Compute the quantiled value."""
        return self.mean_quantile_value / self.len
