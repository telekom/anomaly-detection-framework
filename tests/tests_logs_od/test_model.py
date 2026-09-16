# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from shap import Explanation
from unittest.mock import MagicMock, patch

from adf.logs.models.autoencoder.lstm import LSTMAutoEncoder
from adf.logs.models.autoencoder.model import LogAutoEncoder


class DummyHParams:
    pass


class DummyLogAutoEncoder(LogAutoEncoder):
    @property
    def hparams(self):
        return self.__dict__.setdefault("_hparams", DummyHParams())

    @hparams.setter
    def hparams(self, value):
        self.__dict__["_hparams"] = value

    def save_hyperparameters(self, *args, **kwargs):
        self.hparams = DummyHParams()

    def log(self, name, value, **kwargs):
        self.__dict__.setdefault("logs", {})[name] = value

    def log_dict(self, log_dict, **kwargs):
        self.__dict__.setdefault("log_dicts", {}).update(log_dict)


def identity_forward(self, x, hidden=None, mask=None):
    batch = x.size(0)
    if self.return_hidden:
        dummy = torch.zeros(1, batch, 10, device=x.device)
        return x, (dummy, dummy)
    else:
        return x


@pytest.fixture
def model():
    model_args = {
        "input_dim": 100,
        "hidden_dim": 8,
        "num_layers": 2,
        "dropout": 0.1,
        "batch_first": True,
        "return_hidden": False,
    }

    m = DummyLogAutoEncoder(
        model_name="LSTMAutoEncoder",
        model_args=model_args,
        loss="MSELoss",
        compile_model=False,
        compilation_mode="reduce-overhead",
        optimizer="AdamW",
        optimizer_config={"lr": 1e-4},
        scheduler=None,
        scheduler_config=None,
    )
    m.hparams.compile_model = False
    m.hparams.compilation_mode = "reduce-overhead"

    m.ae.forward = identity_forward.__get__(m.ae, LSTMAutoEncoder)
    return m


def test_initialization(model):
    assert isinstance(model.ae, LSTMAutoEncoder)
    assert callable(model.loss)
    from torchmetrics import MetricCollection

    assert isinstance(model.validation_metrics, MetricCollection)
    assert isinstance(model.test_metrics, MetricCollection)


def test_training_step(model):
    x = torch.randn(2, 5, 100)
    result = model.training_step(x, 0)
    assert "loss" in result


def test_validation_step(model):
    x = torch.randn(2, 5, 100)
    result = model.validation_step(x, 0)
    assert "val_loss" in result and "val_score" in result
    metrics = model.validation_metrics.compute()
    assert any(key.startswith("val_") for key in metrics.keys())


def test_test_step(model):
    x = torch.randn(2, 5, 100)
    result = model.test_step(x, 0)
    assert "test_loss" in result and "test_score" in result
    metrics = model.test_metrics.compute()
    assert any(key.startswith("test_") for key in metrics.keys())


def test_score_method(model):
    x = torch.abs(torch.randn(2, 10, 100))
    score_no_relu = model.score(x, x, add_relu=False, dim=[1, 2])
    score_relu = model.score(x, x, add_relu=True, dim=[1, 2])
    assert score_no_relu.shape == (2,)
    assert score_relu.shape == (2,)
    assert (score_relu >= 0).all()


def test_predict_single(model):
    x = torch.randn(2, 5, 100)
    result = model.predict_single(x, aggregate="all")
    assert result.shape == (2,)
    result = model.predict_single(x, aggregate="embedding")
    assert result.shape == (2, 100)
    result = model.predict_single(x, aggregate="step")
    assert result.shape == (2, 5)


def test_on_fit_start_compile_failure(model, monkeypatch, caplog):
    def dummy_compile(module, mode):
        raise RuntimeError("Compilation failed")

    monkeypatch.setattr(torch, "compile", dummy_compile)
    model.compile_model = True
    model.on_fit_start()
    assert "Failed to compile the model" in caplog.text


def test_on_validation_epoch_end(model):
    model.validation_metrics.update(torch.tensor([0.5, 0.6]))
    if "log_dicts" in model.__dict__:
        del model.__dict__["log_dicts"]
    model.on_validation_epoch_end()
    assert "log_dicts" in model.__dict__
    assert any(key.startswith("val_") for key in model.log_dicts)


def test_on_test_epoch_end(model):
    model.test_metrics.update(torch.tensor([0.7, 0.8]))
    if "log_dicts" in model.__dict__:
        del model.__dict__["log_dicts"]
    model.on_test_epoch_end()
    assert "log_dicts" in model.__dict__
    assert any(key.startswith("test_") for key in model.log_dicts)


def _snapshot(model):
    """Return the values passed to the most recent log_dict call, as plain floats."""
    return {key: float(value) for key, value in model.log_dicts.items()}


def test_on_train_epoch_end_resets_metrics(model):
    model.train_metrics.update(torch.tensor([10.0, 20.0]))
    model.on_train_epoch_end()
    first = _snapshot(model)

    model.train_metrics.update(torch.tensor([0.1, 0.2]))
    model.on_train_epoch_end()
    second = _snapshot(model)

    assert first["train_score_max"] == pytest.approx(20.0)
    assert second["train_score_max"] == pytest.approx(0.2)
    assert second["train_score_mean"] == pytest.approx(0.15)
    # _last_train_score_mean feeds val_error, so it must track the reset metrics too.
    assert model._last_train_score_mean == pytest.approx(0.15)


def test_on_validation_epoch_end_resets_metrics(model):
    model.validation_loss_metric.update(torch.tensor(1.0))
    model.validation_metrics.update(torch.tensor([10.0, 20.0]))
    model.on_validation_epoch_end()
    first = _snapshot(model)

    model.validation_loss_metric.update(torch.tensor(1.0))
    model.validation_metrics.update(torch.tensor([0.1, 0.2]))
    model.on_validation_epoch_end()
    second = _snapshot(model)

    assert first["val_score_max"] == pytest.approx(20.0)
    assert second["val_score_max"] == pytest.approx(0.2)
    assert second["val_score_mean"] == pytest.approx(0.15)
    assert second["val_score_quantile"] == pytest.approx(0.2, abs=1e-3)


def test_on_test_epoch_end_resets_metrics(model):
    model.test_metrics.update(torch.tensor([10.0, 20.0]))
    model.on_test_epoch_end()
    first = _snapshot(model)

    model.test_metrics.update(torch.tensor([0.1, 0.2]))
    model.on_test_epoch_end()
    second = _snapshot(model)

    assert first["test_score_max"] == pytest.approx(20.0)
    assert second["test_score_max"] == pytest.approx(0.2)
    assert second["test_score_mean"] == pytest.approx(0.15)


class DummyOptimizedModule:
    pass


def test_compiled_property(model, monkeypatch):
    model.__dict__["ae"] = object()
    assert not model.is_compiled
    try:
        import torch._dynamo.eval_frame as eval_frame
    except ImportError:

        class DummyEvalFrame:
            OptimizedModule = DummyOptimizedModule

        monkeypatch.setitem(__import__("sys").modules, "torch._dynamo.eval_frame", DummyEvalFrame())
    else:
        monkeypatch.setattr(eval_frame, "OptimizedModule", DummyOptimizedModule)
    model.__dict__["ae"] = DummyOptimizedModule()
    assert model.is_compiled


def test_explain_raises_error_for_non_3d_input():
    model_args = {"input_dim": 100, "hidden_dim": 8, "num_layers": 2}
    model = LogAutoEncoder(model_args=model_args)
    x = torch.randn(10, 100)  # 2D input
    with pytest.raises(ValueError, match="The input tensor must be 3-dimensional"):
        model.explain(x)


def test_explain_raises_error_for_batch_size_greater_than_one():
    model_args = {"input_dim": 100, "hidden_dim": 8, "num_layers": 2}
    model = LogAutoEncoder(model_args=model_args)
    x = torch.randn(2, 5, 100)  # Batch size > 1
    with pytest.raises(ValueError, match="The input tensor must have a batch size of 1"):
        model.explain(x)


def test_explain_computes_shap_values_correctly():
    model_args = {"input_dim": 100, "hidden_dim": 8, "num_layers": 2}
    model = LogAutoEncoder(model_args=model_args)
    x = torch.randn(1, 5, 100)  # Valid input
    with patch("shap.Explainer") as mock_explainer:
        mock_explainer.return_value = MagicMock(return_value=Explanation(values=np.random.randn(5, 100)))
        result = model.explain(x)
        assert isinstance(result, Explanation)
        assert result.values.shape == (5, 100)
