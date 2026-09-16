# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F

from torch import Tensor, nn
from typing_extensions import override

from adf.core.lightning.modules import GaussianNoise

from .lstm import LSTMDecoder, LSTMEncoder
from .registry import add_to_logs_model_registry


class LSTMVariationalReparametrizer(nn.Module):
    """LSTM VAE reparametrizer.

    For input `X`, produce mean and log-variance vectors, and sample latent vector.
    The latent vector is sampled from a multivariate Gaussian with mean `mu`
    and diagonal covariance matrix `diag(exp(logvar))`. States are saved for continuous inference mode.

    """

    def __init__(self, input_dim: int, hidden_dim: int, batch_first: bool):
        """Initialize the LSTM VAE reparametrizer.

        Args:
            input_dim (int) : Input size of tensor
            hidden_dim (int) : Number of hidden units.
            batch_first (bool) : Whether the batch is the first dimension.

        Returns:
            LSTM_VAE (nn.Module) : LSTM VAE module.

        """
        super().__init__()
        self.moduleList = torch.nn.ModuleList()
        h = torch.tensor([], dtype=torch.float32)
        c = torch.tensor([], dtype=torch.float32)

        self.layer_mean = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=batch_first)
        self.layer_logvar = nn.LSTM(input_size=input_dim, hidden_size=input_dim, batch_first=batch_first)

        h = torch.cat((h, torch.tensor([[0] * hidden_dim], dtype=torch.float32)), dim=1)
        c = torch.cat((c, torch.tensor([[0] * hidden_dim], dtype=torch.float32)), dim=1)

        self.register_buffer("h", h, persistent=True)
        self.register_buffer("c", c, persistent=True)

    @staticmethod
    def kl_loss(mu: Tensor, logvar: Tensor) -> Tensor:
        """Compute the KL divergence loss.

        Args:
            mu (Tensor): Mean of the latent Gaussian.
            logvar (Tensor): Log variance of the latent Gaussian.

        Returns:
             Tensor: KL divergence loss.

        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    @staticmethod
    def kl_loss_diagonal(mu: Tensor, logvar: Tensor) -> Tensor:
        """Diagonal KL(q || p) with p ~ N(mu0, diag(exp(logvar0))).

        Here q(z|x) ~ N(mu, diag(exp(logvar))) with logvar = log(sigma^2) per dimension (standard VAE).
        Defaults mu0=0 and logvar0=0 recover the usual prior N(0, I).

        We use ``mean`` (not ``sum``) so the scale is stable across batch/sequence/latent sizes;
        ``LogAutoEncoder`` still applies ``.mean()`` when mixing with reconstruction loss.

        Args:
            mu (Tensor): Mean of the latent Gaussian.
            logvar (Tensor): Log variance of the latent Gaussian (log sigma^2).

        Returns:
            Scalar tensor: mean KL plus a small softplus tail against variance collapse.

        """
        # Reason: match standard Gaussian prior while keeping a numerically safe exp() and a smooth
        # softplus penalty only when logvar drifts very negative (sigma^2 -> 0).
        mu0 = torch.zeros((), device=mu.device, dtype=mu.dtype)
        logvar0 = torch.zeros((), device=logvar.device, dtype=logvar.dtype)
        diff = logvar - logvar0
        exp_term = torch.exp(torch.clamp(diff, min=-30.0, max=30.0))
        kl_per = 0.5 * ((logvar0 - logvar) - 1.0 + exp_term + (mu - mu0).pow(2) * torch.exp(-logvar0))
        kl_mean = kl_per.mean()
        # Softplus is ~0 when logvar >> -12; ramps up when logvar is very negative (collapsed variance).
        collapse_tail = F.softplus(-12.0 - logvar).mean()
        return kl_mean + 1e-4 * collapse_tail

    def sample(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparametrizer trick.

        During the training phase, the reparametrizer computes
        a new state vector by sampling from a multivariate Gaussian distribution.
        During the testing phase, the reparametrizer directly returns the mean.

        Args:
            mu (Tensor): Mean of the latent Gaussian.
            logvar (Tensor): Log variance of the latent Gaussian.

        Returns:
            Tensor: Reparameterized latent vector.

        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.ones_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    @override
    def forward(
        self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = None, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """Perform a forward pass through the LSTM autoencoder.

        Args:
            x (Tensor): The input tensor to the LSTM autoencoder.
            hidden (tuple, optional): A tuple containing the hidden state and cell state
                                      for the LSTM. If None, the default hidden state
                                      (self.h, self.c) is used. Defaults to None.
            mask (Tensor, optional): A tensor containing the mask tensor for ignore timesteps. Defaults to None.

        Returns:
            Tensor: The output tensor from the LSTM autoencoder. If self.hparams.return_hidden
                    is True, returns a list containing the output tensor and a tuple of the
                    concatenated hidden and cell states. Otherwise, returns a list containing
                    only the output tensor.
            Kl divergence loss (Tensor): The KL divergence loss computed during the forward pass.
            hidden (tuple): A tuple containing the hidden state and cell state after the forward pass.

        """
        logvar_hidden = (self.h, self.c)
        logvar_hidden = (
            self.h.expand(x.shape[0], -1).unsqueeze(dim=0).contiguous(),
            self.c.expand(x.shape[0], -1).unsqueeze(dim=0).contiguous(),
        )

        if hidden is None:
            hidden = logvar_hidden

        mu, (h_out, c_out) = self.layer_mean(x, hidden)
        logvar, _ = self.layer_logvar(x, logvar_hidden)

        if self.training:
            out = self.sample(mu, logvar)
        else:
            out = mu

        if mask is not None:
            out *= mask

        kl_loss = self.kl_loss(mu, logvar)
        return out, kl_loss, (h_out, c_out)

    @property
    def dtype(self) -> torch.dtype:
        """Returns the data type (dtype) of the first parameter in the model.

        This method retrieves the data type of the first parameter in the model's parameters iterator.

        Returns:
            torch.dtype: The data type of the first parameter.

        """
        return self.parameters().__next__().dtype


@add_to_logs_model_registry
class LSTMVariationalAutoEncoder(nn.Module):
    """LSTM Variational AutoEncoder model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        batch_first: bool,
        dropout: float,
        return_hidden: bool,
        sigma: float = 0,
        enc_dropout: float = 0,
        dec_dropout: float = 0,
        return_latent: bool = False,
    ):
        """Initialize the LSTM autoencoder.

        Args:
            input_dim (int): The number of expected features in the input.
            hidden_dim (int): The number of features in the hidden state.
            num_layers (int): The number of recurrent layers.
            batch_first (bool, optional): If True, then the input and output tensors are provided as
                (batch, seq, feature). Defaults to True.
            dropout (float, optional): If non-zero, introduces a Dropout layer on the input Defaults to 0.0.
            return_hidden (bool, optional): If True, the hidden states are returned. Defaults to False.
            sigma (float, optional): If non-zero, introduces a Gaussian Noise layer on the input. Defaults to 0.0.
            enc_dropout (float, optional): Dropout rate for the encoder. Defaults to 0.0.
            dec_dropout (float, optional): Dropout rate for the decoder. Defaults to 0.0.
            return_latent (bool, optional): If True, the latent vector is returned. Defaults to False.

        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.gaussian = GaussianNoise(sigma=sigma)

        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=enc_dropout,
            batch_first=batch_first,
        )
        self.reparametrizer = LSTMVariationalReparametrizer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, batch_first=batch_first
        )
        self.decoder = LSTMDecoder(
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dec_dropout,
            batch_first=batch_first,
        )

        self.return_hidden = return_hidden
        self.return_latent = return_latent
        self.register_buffer(
            "h", torch.cat((self.encoder.h, self.reparametrizer.h, self.decoder.h), dim=-1), persistent=True
        )
        self.register_buffer(
            "c", torch.cat((self.encoder.c, self.reparametrizer.c, self.decoder.c), dim=-1), persistent=True
        )

    @override
    def forward(
        self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = None, mask: Tensor | None = None
    ) -> (
        Tensor
        | tuple[Tensor, tuple[Tensor, Tensor]]
        | tuple[Tensor, Tensor]
        | tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]
    ):
        """Perform a forward pass through the LSTM autoencoder.

        Args:
            x (Tensor): The input tensor to the LSTM autoencoder.
            hidden (tuple, optional): A tuple containing the hidden state and cell state
                                      for the LSTM. If None, the default hidden state
                                      (self.h, self.c) is used. Defaults to None.
            mask (Tensor, optional): A tensor containing the mask tensor for ignore timesteps. Defaults to None.

        Returns:
            Tensor: The output tensor and kl_loss from the LSTM variational autoencoder. If self.hparams.return_hidden
                    is True, returns a list containing the output tensor and a tuple of the
                    concatenated hidden and cell states. Otherwise, returns a list containing
                    only the output tensor.

        """
        x = self.dropout(x)
        x = self.gaussian(x)
        if hidden is None:
            hidden = (
                self.h.expand(x.shape[0], -1).unsqueeze(dim=0).contiguous(),
                self.c.expand(x.shape[0], -1).unsqueeze(dim=0).contiguous(),
            )
        h_0, h_1, h_2 = torch.split(
            hidden[0], [self.encoder.h.shape[-1], self.reparametrizer.h.shape[-1], self.decoder.h.shape[-1]], dim=-1
        )
        c_0, c_1, c_2 = torch.split(
            hidden[1], [self.encoder.c.shape[-1], self.reparametrizer.c.shape[-1], self.decoder.c.shape[-1]], dim=-1
        )

        h_0 = h_0.contiguous()
        h_1 = h_1.contiguous()
        h_2 = h_2.contiguous()

        c_0 = c_0.contiguous()
        c_1 = c_1.contiguous()
        c_2 = c_2.contiguous()

        x, (h_0, c_0) = self.encoder(x, (h_0, c_0))
        x_latent, kl_loss, (h_1, c_1) = self.reparametrizer(x, (h_1, c_1))
        x, (h_2, c_2) = self.decoder(x_latent, (h_2, c_2))

        if mask is not None:
            x *= mask

        if self.training:
            if self.return_hidden:
                return (x, kl_loss), (torch.cat((h_0, h_1, h_2), dim=-1), torch.cat((c_0, c_1, c_2), dim=-1))
            else:
                return (x, kl_loss)

        else:
            if self.return_hidden:
                return x, (torch.cat((h_0, h_1, h_2), dim=-1), torch.cat((c_0, c_1, c_2), dim=-1))
            elif self.return_latent:
                return x, c_1  # torch.cat((c_1, h_1), dim=-1)
            else:
                return x

    @property
    def dtype(self) -> torch.dtype:
        """Returns the data type (dtype) of the first parameter in the model.

        This method retrieves the data type of the first parameter in the model's parameters iterator.

        Returns:
            torch.dtype: The data type of the first parameter.

        """
        return self.parameters().__next__().dtype


__all__ = ["LSTMVariationalAutoEncoder"]
