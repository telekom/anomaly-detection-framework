# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import types

from unittest.mock import MagicMock

from adf.logs.models.autoencoder.lstm import LSTMDecoder, LSTMEncoder
from adf.logs.models.autoencoder.model import LogAutoEncoder
from adf.logs.models.autoencoder.model import RegularizedVAE as LogVariationalAutoEncoder
from adf.logs.models.autoencoder.vae import LSTMVariationalAutoEncoder, LSTMVariationalReparametrizer


class DummyHParams:
    pass


class DummyLogAutoEncoder(LogAutoEncoder):
    """Minimal ``LogAutoEncoder`` for unit tests (no Lightning trainer)."""

    @property
    def hparams(self):
        return self.__dict__.setdefault("_hparams", DummyHParams())

    @property
    def model_dtype(self):
        return self.__dict__.setdefault("_model_dtype", torch.float32)

    @hparams.setter
    def hparams(self, value):
        self.__dict__["_hparams"] = value

    def save_hyperparameters(self, *args, **kwargs):
        self.hparams = DummyHParams()

    def log(self, name, value, **kwargs):
        self.__dict__.setdefault("logs", {})[name] = value

    def log_dict(self, log_dict, **kwargs):
        self.__dict__.setdefault("log_dicts", {}).update(log_dict)


class DummyLogVariationalAutoEncoder(LogVariationalAutoEncoder):
    @property
    def hparams(self):
        return self.__dict__.setdefault("_hparams", DummyHParams())

    @property
    def model_dtype(self):
        return self.__dict__.setdefault("_model_dtype", torch.float32)

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
    kl_loss = torch.tensor(1.0, device=x.device)
    if self.return_hidden:
        dummy = torch.zeros(1, batch, 88, device=x.device)
        return x, kl_loss, (dummy, dummy)
    else:
        return x, kl_loss


@pytest.fixture
def model():
    model_args = {
        "input_dim": 10,
        "hidden_dim": 20,
        "num_layers": 2,
        "dropout": 0.1,
        "batch_first": True,
        "return_hidden": False,
    }

    m = DummyLogVariationalAutoEncoder(
        model_name="LSTMVariationalAutoEncoder",
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

    m.ae.forward = identity_forward.__get__(m.ae, LSTMVariationalAutoEncoder)

    return m


def test_initialization(model: DummyLogVariationalAutoEncoder):
    assert isinstance(model.ae, LSTMVariationalAutoEncoder)
    assert callable(model.loss)


def test_forward_no_return_hidden(model: DummyLogVariationalAutoEncoder):
    model.log = MagicMock()
    model.log_dict = MagicMock()
    model.ae.return_hidden = False

    x = torch.randn(2, 5, 100)
    y, _ = model(x)

    # persistent flag
    assert y.shape == x.shape


def test_forward_return_hidden(model: DummyLogVariationalAutoEncoder):
    model.log = MagicMock()
    model.log_dict = MagicMock()
    model.ae.return_hidden = True
    model.return_hidden = True

    x = torch.randn(2, 5, 100)
    y, _, _ = model(x)

    assert y.shape == x.shape


def test_score(model: DummyLogVariationalAutoEncoder):
    x = torch.randn(2, 5, 100)
    y, _ = model(x)
    model.train()
    with torch.enable_grad():
        result = model.score(x, y)
    model.eval()
    with torch.no_grad():
        result_2 = model.score(x, y)
    assert result.shape == ()
    assert result_2.shape == ()


class TestLSTMVariationalAutoEncoder:
    def test_initialization(self):
        autoencoder = LSTMVariationalAutoEncoder(
            input_dim=100, hidden_dim=8, num_layers=2, batch_first=True, dropout=0.1, return_hidden=True
        )
        assert isinstance(autoencoder.encoder, LSTMEncoder)
        assert isinstance(autoencoder.decoder, LSTMDecoder)
        assert autoencoder.h.shape == (1, 88)
        assert autoencoder.c.shape == (1, 88)
        assert autoencoder.return_hidden is True

    def test_forward_return_hidden(self):
        autoencoder = LSTMVariationalAutoEncoder(
            input_dim=100, hidden_dim=8, num_layers=2, batch_first=True, dropout=0.1, return_hidden=True
        )
        x = torch.randn(2, 5, 100)
        (out, kl_loss), hidden = autoencoder(x)
        (out, kl_loss), hidden_2 = autoencoder(x, hidden=hidden)
        assert out.shape == (2, 5, 100)
        assert kl_loss.shape == ()
        assert hidden[0].shape == (1, 2, 88)
        assert hidden[1].shape == (1, 2, 88)
        assert hidden_2[0].shape == (1, 2, 88)
        assert hidden_2[1].shape == (1, 2, 88)

    def test_forward_not_return_hidden(self):
        autoencoder = LSTMVariationalAutoEncoder(
            input_dim=100, hidden_dim=8, num_layers=2, batch_first=True, dropout=0.1, return_hidden=False
        )
        x = torch.randn(2, 5, 100)
        (out, kl_loss) = autoencoder(x)
        assert out.shape == (2, 5, 100)
        assert kl_loss.shape == ()

    def test_dtype_property(self):
        autoencoder = LSTMVariationalAutoEncoder(
            input_dim=100, hidden_dim=8, num_layers=2, batch_first=True, dropout=0.1, return_hidden=True
        )
        first_param_dtype = next(autoencoder.parameters()).dtype
        assert autoencoder.dtype == first_param_dtype
        autoencoder_double = autoencoder.double()
        assert autoencoder_double.dtype == torch.float64


def _reference_kl_sum_unit_prior(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Same math as ``LSTMVariationalReparametrizer.kl_loss`` (standard VAE KL, summed)."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class TestKLLoss:
    def test_kl_matches_reference_implementation(self):
        torch.manual_seed(0)
        mu = torch.randn(2, 5, 8)
        logvar = torch.randn(2, 5, 8) * 0.5
        got = LSTMVariationalReparametrizer.kl_loss(mu, logvar)
        ref = _reference_kl_sum_unit_prior(mu, logvar)
        torch.testing.assert_close(got, ref)

    def test_kl_zero_when_posterior_equals_prior(self):
        mu = torch.zeros(2, 3, 4)
        logvar = torch.zeros(2, 3, 4)
        kl = LSTMVariationalReparametrizer.kl_loss(mu, logvar)
        assert kl.shape == ()
        torch.testing.assert_close(kl, torch.tensor(0.0))

    def test_kl_classical_form_summed(self):
        """``kl_loss`` sums 0.5*(exp(logvar) + mu^2 - 1 - logvar) over all latent dims."""
        mu = torch.randn(3, 4, 6)
        logvar = torch.clamp(torch.randn(3, 4, 6) * 0.2, -5.0, 5.0)
        classical = 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar)
        expected = classical.sum()
        got = LSTMVariationalReparametrizer.kl_loss(mu, logvar)
        torch.testing.assert_close(got, expected)


class TestLogAutoEncoderKlScaling:
    def test_training_step_scales_kl_with_detach_ratio(self):
        """``training_step`` adds ``beta * (recon.detach() / (kl.detach()+eps)) * kl`` (see ``LogAutoEncoder``)."""
        model_args = {
            "input_dim": 10,
            "hidden_dim": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "batch_first": True,
            "return_hidden": False,
        }
        model = DummyLogAutoEncoder(
            model_name="LSTMVariationalAutoEncoder",
            model_args=model_args,
            loss="MSELoss",
            compile_model=False,
            compilation_mode="reduce-overhead",
            optimizer="AdamW",
            optimizer_config={"lr": 1e-4},
            scheduler=None,
            scheduler_config=None,
            use_reg_loss=True,
            beta=0.5,
        )
        model.hparams.compile_model = False
        model.log = MagicMock()

        def mock_ae_forward(self, x, hidden=None, mask=None):
            x_hat = x * 0.25 + 0.5
            kl = torch.tensor(4.0, device=x.device, dtype=x.dtype)
            return x_hat, kl

        model.ae.forward = types.MethodType(mock_ae_forward, model.ae)  # type: ignore[method-assign]

        model.train()
        batch = torch.ones(2, 3, 10) * 2.0
        x_hat, kl = mock_ae_forward(model.ae, batch)
        recon = model.score(x_hat, batch, add_relu=False, mask=None)
        scale = recon.detach() / (kl.detach().mean() + 1e-8)
        expected_loss = recon + model.beta * scale * kl.mean()

        out = model.training_step(batch, 0)
        torch.testing.assert_close(out["loss"], expected_loss)
