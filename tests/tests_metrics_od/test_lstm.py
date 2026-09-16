# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import torch

from adf.metrics.models.autoencoder.lstm import LSTMAutoEncoder, LSTMDecoder, LSTMEncoder, TimeEncoder


def test_time_encoder_forward():
    encoder = TimeEncoder(time_input_size=6, hidden_size=4, dropout=0.0)
    x = torch.rand(10, 6)
    out = encoder(x)
    assert out.shape == (10, 4)


def test_lstm_encoder_forward_without_hidden():
    batch_size = 5
    seq_len = 7
    input_size = 12
    hidden_size = 32
    num_layers = 2
    encoder = LSTMEncoder(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.0, batch_first=True
    )
    x = torch.rand(batch_size, seq_len, input_size)
    output, (h_n, c_n) = encoder(x)
    assert output.shape == (batch_size, seq_len, hidden_size * 2)
    assert h_n.shape == (2 * num_layers, batch_size, hidden_size)
    assert c_n.shape == (2 * num_layers, batch_size, hidden_size)


def test_lstm_encoder_forward_with_hidden():
    batch_size = 3
    seq_len = 4
    input_size = 10
    hidden_size = 16
    num_layers = 1
    encoder = LSTMEncoder(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.0, batch_first=True
    )
    x = torch.rand(batch_size, seq_len, input_size)
    h_init = torch.rand(2 * num_layers, batch_size, hidden_size)
    c_init = torch.rand(2 * num_layers, batch_size, hidden_size)
    output, (h_n, c_n) = encoder(x, hidden=(h_init, c_init))
    assert output.shape == (batch_size, seq_len, hidden_size * 2)
    assert h_n.shape == (2 * num_layers, batch_size, hidden_size)
    assert c_n.shape == (2 * num_layers, batch_size, hidden_size)


def test_lstm_decoder_forward_without_hidden():
    batch_size = 4
    seq_len = 5
    hidden_size = 32
    num_layers = 1
    output_size = 8
    decoder = LSTMDecoder(
        output_size=output_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.0, batch_first=True
    )
    x = torch.rand(batch_size, seq_len, hidden_size * 2)
    out, (h_n, c_n) = decoder(x)
    assert out.shape == (batch_size, seq_len, output_size)
    assert h_n.shape == (2 * num_layers, batch_size, hidden_size)
    assert c_n.shape == (2 * num_layers, batch_size, hidden_size)


def test_lstm_decoder_forward_with_hidden():
    batch_size = 3
    seq_len = 6
    hidden_size = 16
    num_layers = 2
    output_size = 10
    decoder = LSTMDecoder(
        output_size=output_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.0, batch_first=True
    )
    x = torch.rand(batch_size, seq_len, hidden_size * 2)
    h_init = torch.rand(2 * num_layers, batch_size, hidden_size)
    c_init = torch.rand(2 * num_layers, batch_size, hidden_size)
    out, (h_n, c_n) = decoder(x, hidden=(h_init, c_init))
    assert out.shape == (batch_size, seq_len, output_size)
    assert h_n.shape == (2 * num_layers, batch_size, hidden_size)
    assert c_n.shape == (2 * num_layers, batch_size, hidden_size)


def test_lstm_autoencoder_forward_without_hidden():
    batch_size = 4
    seq_len = 7
    input_dim = 10
    hidden_dim = 32
    autoencoder = LSTMAutoEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        time_dim=6,
        num_layers=1,
        dropout=0.0,
        batch_first=True,
        return_hidden=False,
    )
    x = torch.rand(batch_size, seq_len, input_dim)
    decoded, metrics = autoencoder(x)
    assert decoded.shape == (batch_size, seq_len, input_dim - 6)
    expected_metrics = x[:, :, : input_dim - 6]
    assert torch.allclose(metrics, expected_metrics)


def test_lstm_autoencoder_forward_with_hidden():
    batch_size = 3
    seq_len = 5
    input_dim = 12
    hidden_dim = 16
    num_layers = 2
    autoencoder = LSTMAutoEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        time_dim=6,
        num_layers=num_layers,
        dropout=0.0,
        batch_first=True,
        return_hidden=True,
    )
    x = torch.rand(batch_size, seq_len, input_dim)
    decoded, metrics, hidden_states = autoencoder(x)
    assert decoded.shape == (batch_size, seq_len, input_dim - 6)
    expected_metrics = x[:, :, : input_dim - 6]
    assert torch.allclose(metrics, expected_metrics)
    h_n, c_n = hidden_states
    assert h_n.shape == (2 * num_layers, batch_size, hidden_dim)
    assert c_n.shape == (2 * num_layers, batch_size, hidden_dim)


def test_lstm_autoencoder_buffers():
    input_dim = 12
    hidden_dim = 16
    num_layers = 2
    autoencoder = LSTMAutoEncoder(
        input_dim=input_dim, hidden_dim=hidden_dim, time_dim=6, num_layers=num_layers, dropout=0.0, batch_first=True
    )
    expected_shape = (2 * num_layers, 1, hidden_dim * 2)
    assert autoencoder.h.shape == expected_shape
    assert autoencoder.c.shape == expected_shape
