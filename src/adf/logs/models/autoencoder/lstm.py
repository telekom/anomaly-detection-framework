# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from torch import Tensor, nn
from typing_extensions import override

from adf.core.lightning.modules import GaussianNoise

from .registry import add_to_logs_model_registry


class LSTMEncoder(nn.Module):
    """Encoder part.

    States are saved for continuous inference mode.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, batch_first: bool):
        """Initialize the LSTM Encoder.

        Args:
            input_dim (int) : Input size of tensor
            hidden_dim (int) : Number of hidden units.
            num_layers (int) : Number of stacked layers that will be created.
            dropout (float) : Dropout ratio.
            batch_first (bool) : Whether the batch is the first dimension.

        Returns:
            LSTM_Encoder (nn.Module) : LSTM Encoder module.

        """
        super().__init__()

        self.moduleList = torch.nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        current_size = input_dim
        h = torch.tensor([], dtype=torch.float32)
        c = torch.tensor([], dtype=torch.float32)
        h_out = torch.tensor([], dtype=torch.float32)
        c_out = torch.tensor([], dtype=torch.float32)
        slice_list = []
        old_size = 0

        for i in range(1, num_layers + 1):
            self.moduleList.append(
                nn.LSTM(
                    input_size=current_size,
                    hidden_size=hidden_dim * (4 ** (num_layers - i)),
                    dropout=0.0,  # Only for intermediate layers
                    batch_first=batch_first,
                )
            )
            self.dropout_layers.append(nn.Dropout(dropout))
            current_size = hidden_dim * (4 ** (num_layers - i))
            h = torch.cat((h, torch.tensor([[0] * current_size], dtype=torch.float32)), dim=1)
            c = torch.cat((c, torch.tensor([[0] * current_size], dtype=torch.float32)), dim=1)
            slice_list.append([old_size, old_size + current_size])
            old_size += current_size

        self.register_buffer("h", h, persistent=True)
        self.register_buffer("c", c, persistent=True)
        self.register_buffer("slice_list", torch.tensor(slice_list), persistent=True)
        self.register_buffer("h_out", h_out, persistent=True)
        self.register_buffer("c_out", c_out, persistent=True)

    @override
    def forward(self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch_size = x.size()[0]
        if hidden is None:
            (h_0, c_0) = (
                self.h.expand(batch_size, -1).unsqueeze(dim=0),
                self.c.expand(batch_size, -1).unsqueeze(dim=0),
            )
        else:
            (h_0, c_0) = (hidden[0], hidden[1])

        h_out = self.h_out
        c_out = self.c_out

        for i, layer in enumerate(self.moduleList):
            h = h_0[..., self.slice_list[i][0] : self.slice_list[i][1]].contiguous()
            c = c_0[..., self.slice_list[i][0] : self.slice_list[i][1]].contiguous()
            x, (h, c) = layer(x, (h, c))

            if i != len(self.moduleList) - 1:
                x = self.dropout_layers[i](x)

            h_out = torch.cat((h_out, h), dim=-1)
            c_out = torch.cat((c_out, c), dim=-1)

        return x, (h_out, c_out)


class LSTMDecoder(nn.Module):
    """Decoder part. States are saved for continuous inference mode."""

    def __init__(self, output_dim: int, hidden_dim: int, num_layers: int, dropout: float, batch_first: bool):
        """Initialize the LSTM decoder.

        Args:
            output_dim (int) : Input size of tensor
            hidden_dim (int) : Number of hidden units.
            num_layers (int) : Number of stacked layers that will be created.
            dropout (float) : Dropout ratio.
            batch_first (bool) : Whether the batch is the first dimension.

        Returns:
            LSTM_Encoder (nn.Module) : LSTM Encoder module.

        """
        super().__init__()

        self.moduleList = torch.nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        current_size = hidden_dim
        h = torch.tensor([], dtype=torch.float32)
        c = torch.tensor([], dtype=torch.float32)
        h_out = torch.tensor([], dtype=torch.float32)
        c_out = torch.tensor([], dtype=torch.float32)

        slice_list = []
        old_size = 0

        for i in range(0, num_layers):
            self.moduleList.append(
                nn.LSTM(
                    input_size=current_size,
                    hidden_size=hidden_dim * (4**i),
                    dropout=0.0,  # Only for intermediate layers
                    batch_first=batch_first,
                )
            )
            self.dropout_layers.append(nn.Dropout(dropout))

            current_size = hidden_dim * (4**i)
            h = torch.cat((h, torch.tensor([[0] * current_size], dtype=torch.float32)), dim=1)
            c = torch.cat((c, torch.tensor([[0] * current_size], dtype=torch.float32)), dim=1)
            slice_list.append([old_size, old_size + current_size])
            old_size += current_size
        self.lin = nn.Linear(current_size, output_dim)
        self.register_buffer("h", h, persistent=True)
        self.register_buffer("c", c, persistent=True)
        self.register_buffer("slice_list", torch.tensor(slice_list), persistent=True)
        self.register_buffer("h_out", h_out, persistent=True)
        self.register_buffer("c_out", c_out, persistent=True)

    @override
    def forward(self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch_size = x.size()[0]

        if hidden is None:
            (h_0, c_0) = (
                self.h.expand(batch_size, -1).unsqueeze(dim=0),
                self.c.expand(batch_size, -1).unsqueeze(dim=0),
            )
        else:
            (h_0, c_0) = (hidden[0], hidden[1])

        h_out = self.h_out
        c_out = self.c_out
        for i, layer in enumerate(self.moduleList):
            h = h_0[..., self.slice_list[i][0] : self.slice_list[i][1]].contiguous()
            c = c_0[..., self.slice_list[i][0] : self.slice_list[i][1]].contiguous()
            x, (h, c) = layer(x, (h, c))
            x = self.dropout_layers[i](x)
            h_out = torch.cat((h_out, h), dim=-1)
            c_out = torch.cat((c_out, c), dim=-1)
        x = self.lin(x)
        return x, (h_out, c_out)


@add_to_logs_model_registry
class LSTMAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        num_layers: int = 2,
        batch_first: bool = True,
        return_hidden: bool = False,
        dropout: float = 0.2,
        enc_dropout: float = 0.0,
        dec_dropout: float = 0.0,
        sigma: float = 0,
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
            enc_dropout (float, optional): If non-zero, introduces a Dropout layer on the LSTM encoder. Defaults to 0.0.
            dec_dropout (float, optional): If non-zero, introduces a Dropout layer on the LSTM decoder. Defaults to 0.0.
            sigma (float, optional): Standard deviation for Gaussian noise added to the input tensor. Defaults to 0.0.
            return_latent (bool, optional): If True, the latent vector is returned. Defaults to False.

        """
        super().__init__()
        if hidden_dim & (hidden_dim - 1) != 0 or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a power of 2")

        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=enc_dropout,
            batch_first=batch_first,
        )
        self.decoder = LSTMDecoder(
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dec_dropout,
            batch_first=batch_first,
        )

        self.return_hidden = return_hidden
        self.register_buffer("h", torch.cat((self.encoder.h, self.decoder.h), dim=-1), persistent=True)
        self.register_buffer("c", torch.cat((self.encoder.c, self.decoder.c), dim=-1), persistent=True)

        self.dropout = nn.Dropout(dropout)
        self.gaussian = GaussianNoise(sigma=sigma)
        self.return_latent = return_latent

    @override
    def forward(
        self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = None, mask: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, tuple[Tensor, Tensor]]:
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

        """
        x = self.dropout(x)
        x = self.gaussian(x)

        if hidden is None:
            hidden = (self.h, self.c)
            hidden = (
                self.h.expand(x.shape[0], -1).unsqueeze(dim=0).contiguous(),
                self.c.expand(x.shape[0], -1).unsqueeze(dim=0).contiguous(),
            )
        h_0, h_1 = torch.split(hidden[0], [self.encoder.h.shape[-1], self.decoder.h.shape[-1]], dim=-1)
        c_0, c_1 = torch.split(hidden[1], [self.encoder.c.shape[-1], self.decoder.c.shape[-1]], dim=-1)
        x, (h_0, c_0) = self.encoder(x, (h_0, c_0))
        x, (h_1, c_1) = self.decoder(x, (h_1, c_1))

        if mask is not None:
            x *= mask

        if self.return_hidden:
            return x, (torch.cat((h_0, h_1), dim=-1), torch.cat((c_0, c_1), dim=-1))
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
