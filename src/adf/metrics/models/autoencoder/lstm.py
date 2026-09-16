# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import torch

from torch import Tensor, nn
from typing_extensions import override

from adf.core.lightning.modules import GaussianNoise


class TimeEncoder(nn.Module):
    """Simple feed-forward encoder for time features."""

    def __init__(self, time_input_size: int = 6, hidden_size: int = 4, dropout: float = 0.1):
        """Initialize the TimeEncoder module.

        Args:
            time_input_size (int) : Number of time features
            hidden_size (int) : Number of hidden units.
            dropout (float) : Dropout ratio.

        Returns:
            LSTM_Encoder (nn.Module) : LSTM Time Encoder module.

        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(time_input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass through the model.

        Args:
            x (Tensor): Input tensor to be passed through the encoder.

        Returns:
            Tensor: Output tensor after being processed by the encoder.

        """
        # We can't typecheck nn.Sequential, so we need to ignore the type here
        return self.encoder.forward(x)  # type: ignore


class LSTMEncoder(nn.Module):
    """Encoder part.

    States are saved for continuous inference mode.
    """

    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        """Initialize the LSTM Encoder.

        Args:
            input_size (int) : Input size of tensor
            hidden_size (int) : Number of hidden units.
            num_layers (int) : Number of stacked layers that will be created.
            dropout (float) : Dropout ratio.
            batch_first (bool) : Whether the batch is the first dimension.

        Returns:
            LSTM_Encoder output, hidden_state, ......

        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first,
            bidirectional=True,
        )

        # Initialize hidden states
        h = torch.zeros(2 * num_layers, 1, hidden_size)  # 2 for bidirectional
        c = torch.zeros(2 * num_layers, 1, hidden_size)

        self.register_buffer("h", h, persistent=True)
        self.register_buffer("c", c, persistent=True)

    @override
    def forward(self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Perform a forward pass through the LSTM model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
            hidden (tuple[Tensor, Tensor] | None, optional):
                A tuple containing the initial hidden state (h) and cell state (c) for the LSTM.
                Each tensor in the tuple should have shape (2 * num_layers, batch_size, hidden_size).
                If None, the hidden state and cell state will be initialized to zeros. Defaults to None.

        Returns:
            tuple[Tensor, tuple[Tensor, Tensor]]: A tuple containing:
                - output (Tensor): The output tensor of shape (batch_size, sequence_length, hidden_size).
                - (h_n, c_n) (tuple[Tensor, Tensor]): A tuple containing the final hidden and cell state, each of shape
                  (2 * num_layers, batch_size, hidden_size).

        """
        batch_size = x.size(0)
        if hidden is None:
            h = self.h.expand(-1, batch_size, -1).contiguous()
            c = self.c.expand(-1, batch_size, -1).contiguous()
        else:
            h, c = hidden
            h = h.view(2 * self.num_layers, batch_size, self.hidden_size)
            c = c.view(2 * self.num_layers, batch_size, self.hidden_size)

        output, (h_n, c_n) = self.lstm(x, (h, c))
        return output, (h_n, c_n)


class LSTMDecoder(nn.Module):
    """Decoder part.

    States are saved for continuous inference mode.
    """

    def __init__(
        self,
        output_size: int = 8,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        """Initialize the LSTM Decoder.

        Args:
            output_size (int) : output size of the tensor
            hidden_size (int) : Number of hidden units.
            num_layers (int) : Number of stacked layers that will be created.
            dropout (float) : Dropout ratio.
            batch_first (bool) : Whether the batch is the first dimension.

        Returns:
            LSTM Decoder output and hidden states

        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Bi-LSTM Model
        self.lstm = nn.LSTM(
            input_size=hidden_size * 2,  # bidirectional input
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first,
            bidirectional=True,
        )

        self.linear = nn.Linear(hidden_size * 2, output_size)

        h = torch.zeros(2 * num_layers, 1, hidden_size)
        c = torch.zeros(2 * num_layers, 1, hidden_size)

        self.register_buffer("h", h, persistent=True)
        self.register_buffer("c", c, persistent=True)

    @override
    def forward(self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Perform a forward pass through the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
            hidden (tuple[Tensor, Tensor] | None, optional): A tuple containing the hidden state and cell state
                tensors of shape (num_layers * 2, batch_size, hidden_size). If None, the hidden state and cell
                state are initialized to zeros. Defaults to None.

        Returns:
            tuple[Tensor, tuple[Tensor, Tensor]]: A tuple containing:
                - output (Tensor): The output tensor of shape (batch_size, sequence_length, output_size).
                - (h_n, c_n) (tuple[Tensor, Tensor]): A tuple containing the final hidden state and cell state
                  tensors of shape (num_layers * 2, batch_size, hidden_size).

        """
        batch_size = x.size(0)
        if hidden is None:
            h = self.h.expand(-1, batch_size, -1).contiguous()
            c = self.c.expand(-1, batch_size, -1).contiguous()
        else:
            h, c = hidden
            h = h.view(2 * self.num_layers, batch_size, self.hidden_size)
            c = c.view(2 * self.num_layers, batch_size, self.hidden_size)

        output, (h_n, c_n) = self.lstm(x, (h, c))
        output = self.linear(output)
        return output, (h_n, c_n)


class LSTMAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        time_dim: int = 6,
        num_layers: int = 1,
        dropout: float = 0,
        enc_dropout: float | None = None,
        dec_dropout: float | None = None,
        sigma: float = 0.0,
        batch_first: bool = True,
        return_hidden: bool = False,
    ):
        """Initialize the LSTM Autoencoder.

        Args:
            input_dim (int) : Input size of tensor.
            hidden_dim (int) : Number of hidden units.
            time_dim (int) : Number of time features.
            num_layers (int) : Number of stacked layers that will be created.
            dropout (float) : Dropout applied to metrics features before encoding. Also used as
                fallback for enc_dropout/dec_dropout/TimeEncoder dropout when they are None.
            enc_dropout (float | None) : Dropout for LSTM encoder inter-layer dropout.
                Falls back to ``dropout`` when None.
            dec_dropout (float | None) : Dropout for LSTM decoder inter-layer dropout.
                Falls back to ``dropout`` when None.
            sigma (float) : Standard deviation for multiplicative Gaussian noise on metrics features.
                Only active during training. Defaults to 0.0 (no noise).
            batch_first (bool) : Whether the batch is the first dimension.
            return_hidden (bool) : Whether to return the hidden states.

        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_metrics = input_dim - time_dim
        self.return_hidden = return_hidden

        enc_dropout = enc_dropout if enc_dropout is not None else dropout
        dec_dropout = dec_dropout if dec_dropout is not None else dropout

        self.input_dropout = nn.Dropout(dropout)
        self.gaussian_noise = GaussianNoise(sigma=sigma)

        self.time_encoder = TimeEncoder(time_input_size=time_dim, hidden_size=4, dropout=dropout)

        self.encoder = LSTMEncoder(
            input_size=self.n_metrics + 4,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=enc_dropout,
            batch_first=batch_first,
        )

        self.decoder = LSTMDecoder(
            output_size=self.n_metrics,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dec_dropout,
            batch_first=batch_first,
        )

        self.register_buffer("h", torch.cat((self.encoder.h, self.decoder.h), dim=-1), persistent=True)
        self.register_buffer("c", torch.cat((self.encoder.c, self.decoder.c), dim=-1), persistent=True)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """Perform a forward pass through the LSTM autoencoder.

        Args:
            x (Tensor): The input tensor to the LSTM autoencoder.

        Returns:
            Tensor: The output tensor from the LSTM autoencoder. If return_hidden
                    is True, returns the output tensor and a tuple of the
                    concatenated hidden and cell states. Otherwise, returns
                    the output tensor and metrics.

        """
        metrics = x[:, :, : self.n_metrics]
        time_features = x[:, :, self.n_metrics :]

        # Regularization applied only to metrics features, not time.
        # Keep original metrics as the reconstruction target.
        metrics_noisy = self.input_dropout(metrics)
        metrics_noisy = self.gaussian_noise(metrics_noisy)

        encoded_time = self.time_encoder(time_features)
        combined = torch.cat([metrics_noisy, encoded_time], dim=-1)

        encoded, (h_n, c_n) = self.encoder(combined)
        decoded, _ = self.decoder(encoded)

        if self.return_hidden:
            return decoded, metrics, (h_n, c_n)
        return decoded, metrics

    @property
    def dtype(self) -> torch.dtype:
        """Returns the data type (dtype) of the first parameter in the model.

        This method retrieves the data type of the first parameter in the model's parameters iterator.

        Returns:
            torch.dtype: The data type of the first parameter.

        """
        return self.parameters().__next__().dtype
