# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from src.adf.core.lightning.metrics import MeanQuantileMetric


def computes_correct_quantile_for_tensor():
    metric = MeanQuantileMetric(quantile=0.5)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    metric.update(values)
    result = metric.compute()
    assert torch.isclose(result, torch.tensor(3.0))


def handles_nan_values_with_warn_strategy(caplog):
    metric = MeanQuantileMetric(quantile=0.5, nan_strategy="warn")
    values = torch.tensor([1.0, float("nan"), 3.0])
    metric.update(values)
    result = metric.compute()
    assert "NaN values encountered" in caplog.text
    assert torch.isclose(result, torch.tensor(2.0))


def raises_error_for_nan_values_with_error_strategy():
    metric = MeanQuantileMetric(quantile=0.5, nan_strategy="error")
    values = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(RuntimeError, match="NaN values encountered"):
        metric.update(values)


def imputes_nan_values_with_float_strategy():
    metric = MeanQuantileMetric(quantile=0.5, nan_strategy=0.0)
    values = torch.tensor([1.0, float("nan"), 3.0])
    metric.update(values)
    result = metric.compute()
    assert torch.isclose(result, torch.tensor(2.0))


def computes_correct_quantile_for_multiple_updates():
    metric = MeanQuantileMetric(quantile=0.5)
    metric.update(torch.tensor([1.0, 2.0, 3.0]))
    metric.update(torch.tensor([4.0, 5.0, 6.0]))
    result = metric.compute()
    assert torch.isclose(result, torch.tensor(3.5))


def handles_empty_tensor_update():
    metric = MeanQuantileMetric(quantile=0.5)
    metric.update(torch.tensor([]))
    result = metric.compute()
    assert result == 0.0
