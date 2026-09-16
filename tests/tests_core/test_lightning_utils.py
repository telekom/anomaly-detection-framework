# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from torch import nn, optim
from torch.nn.functional import pad
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from adf.core.lightning.utils import construct_loss, construct_optimizer, construct_scheduler, reduce


def test_reduce_no_mask():
    x = torch.abs(torch.randn(2, 10, 100))
    result = reduce(x, reduction="mean")
    assert result.shape == ()
    result = reduce(x, reduction="mean", dim=[1, 2])
    assert result.shape == (2,)
    result = reduce(x, reduction="mean", dim=-1)
    assert result.shape == (2, 10)


def test_reduce_with_mask():
    x = torch.abs(torch.randn(2, 10, 100))
    mask = pad(torch.ones(2, 5, 100), (0, 0, 0, 5))
    result_no_mask = reduce(x * mask, reduction="mean")
    result_mask = reduce(x * mask, reduction="mean", mask=mask)

    assert result_no_mask.shape == ()
    assert result_mask.shape == ()
    assert result_no_mask < result_mask  # averaging over all vs valid only


def test_construct_optimizer():
    params = [nn.Parameter(torch.randn(2, 2))]
    optim_name = "Adam"
    optim_config = {"lr": 0.001}
    optimizer = construct_optimizer(params, optim_name, optim_config)
    assert isinstance(optimizer, optim.Adam)

    with pytest.raises(ValueError):
        construct_optimizer(params, "InvalidOptimizer", optim_config)


def test_construct_scheduler():
    params = [nn.Parameter(torch.randn(2, 2))]
    optimizer = optim.Adam(params, lr=0.001)

    scheduler_name = "StepLR"
    scheduler_config = {"step_size": 10, "gamma": 0.1}
    scheduler = construct_scheduler(optimizer, scheduler_name, scheduler_config)
    assert isinstance(scheduler, StepLR)

    scheduler_name = "ReduceLROnPlateau"
    scheduler_config = {"mode": "min", "factor": 0.1, "patience": 10}
    scheduler = construct_scheduler(optimizer, scheduler_name, scheduler_config)
    assert isinstance(scheduler, ReduceLROnPlateau)

    with pytest.raises(ValueError):
        construct_scheduler(optimizer, "InvalidScheduler", scheduler_config)


def test_construct_loss():
    loss_name = "CrossEntropyLoss"
    loss_kwargs = {}
    loss = construct_loss(loss_name, loss_kwargs)
    assert isinstance(loss, nn.CrossEntropyLoss)

    loss_name = "MSELoss"
    loss_kwargs = {}
    loss = construct_loss(loss_name, loss_kwargs)
    assert isinstance(loss, nn.MSELoss)

    with pytest.raises(ValueError):
        construct_loss("InvalidLoss", loss_kwargs)
