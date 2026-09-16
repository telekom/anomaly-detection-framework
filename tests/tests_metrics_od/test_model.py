# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from adf.metrics.models.autoencoder.model import MetricsAutoEncoder


@pytest.fixture
def model():
    m = MetricsAutoEncoder(
        input_dim=10,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        batch_first=True,
        return_hidden=False,
        loss="MSELoss",
        optimizer="AdamW",
    )
    m.log = lambda *args, **kwargs: None
    m.log_dict = lambda d, **kwargs: setattr(m, "last_log_dict", d)
    return m


def test_forward_shape(model):
    batch = torch.rand(8, 20, 10)
    decoded, metrics = model.forward(batch)
    assert decoded.shape == (8, 20, 4)
    assert metrics.shape == (8, 20, 4)


def test_score_function(model):
    x = torch.ones(4, 5, 4)
    y = torch.ones(4, 5, 4)
    score = model.score(x, y)
    assert torch.allclose(score, torch.zeros(4))


def test_training_step(model):
    batch = torch.rand(8, 20, 10)
    output = model.training_step(batch, 0)
    assert "loss" in output
    loss = output["loss"]
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_validation_step(model):
    batch = torch.rand(8, 20, 10)
    output = model.validation_step(batch, 0)
    assert "val_loss" in output
    assert "val_score" in output
    computed = model.validation_metrics.compute()
    for key in ["val_score_max", "val_score_mean", "val_score_quantile"]:
        assert key in computed


def test_test_step(model):
    batch = torch.rand(8, 20, 10)
    output = model.test_step(batch, 0)
    assert "test_loss" in output
    assert "test_score" in output
    computed = model.test_metrics.compute()
    for key in ["test_score_max", "test_score_mean", "test_score_quantile"]:
        assert key in computed


def test_predict_step(model):
    batch = torch.rand(8, 20, 10)
    pred = model.predict_step(batch, 0)
    assert pred.shape == (8,)


def test_on_validation_epoch_end(model):
    batch = torch.rand(8, 20, 10)
    _ = model.validation_step(batch, 0)
    if hasattr(model, "last_log_dict"):
        del model.last_log_dict
    model.on_validation_epoch_end()
    assert hasattr(model, "last_log_dict")
    last_log = model.last_log_dict
    for key in ["val_score_max", "val_score_mean", "val_score_quantile"]:
        assert key in last_log


def test_on_test_epoch_end(model):
    batch = torch.rand(8, 20, 10)
    _ = model.test_step(batch, 0)
    if hasattr(model, "last_log_dict"):
        del model.last_log_dict
    model.on_test_epoch_end()
    assert hasattr(model, "last_log_dict")
    last_log = model.last_log_dict
    for key in ["test_score_max", "test_score_mean", "test_score_quantile"]:
        assert key in last_log
