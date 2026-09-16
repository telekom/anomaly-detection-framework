# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import torch

from adf.logs.models.autoencoder.lstm import LSTMAutoEncoder, LSTMDecoder, LSTMEncoder


class TestLSTMEncoder:
    def test_initialization(self):
        encoder = LSTMEncoder(input_dim=100, hidden_dim=8, num_layers=2, dropout=0.1, batch_first=True)
        assert len(encoder.moduleList) == 2
        assert len(encoder.dropout_layers) == 2

        assert encoder.moduleList[0].input_size == 100
        assert encoder.moduleList[0].hidden_size == 32
        assert encoder.moduleList[1].input_size == 32
        assert encoder.moduleList[1].hidden_size == 8

        expected_slices = torch.tensor([[0, 32], [32, 40]])
        torch.testing.assert_close(encoder.slice_list, expected_slices)

        assert encoder.h.shape == (1, 40)
        assert encoder.c.shape == (1, 40)

    def test_forward_batch_first(self):
        encoder = LSTMEncoder(input_dim=100, hidden_dim=8, num_layers=2, dropout=0.1, batch_first=True)
        x = torch.randn(2, 5, 100)
        out, hidden = encoder(x)
        assert out.shape == (2, 5, 8)
        assert hidden[0].shape == (1, 2, 40)
        assert hidden[1].shape == (1, 2, 40)

    def test_forward_batch_first_with_hidden(self):
        encoder = LSTMEncoder(input_dim=100, hidden_dim=8, num_layers=2, dropout=0.1, batch_first=True)
        hidden_in = (encoder.h.expand(2, -1).unsqueeze(dim=0), encoder.c.expand(2, -1).unsqueeze(dim=0))
        x = torch.randn(2, 5, 100)
        out, hidden = encoder(x, hidden_in)
        assert out.shape == (2, 5, 8)
        assert hidden[0].shape == (1, 2, 40)
        assert hidden[1].shape == (1, 2, 40)


class TestLSTMDecoder:
    def test_initialization(self):
        decoder = LSTMDecoder(output_dim=100, hidden_dim=8, num_layers=2, dropout=0.1, batch_first=True)
        assert len(decoder.moduleList) == 2
        assert len(decoder.dropout_layers) == 2

        assert decoder.moduleList[0].input_size == 8
        assert decoder.moduleList[0].hidden_size == 8
        assert decoder.moduleList[1].input_size == 8
        assert decoder.moduleList[1].hidden_size == 32

        expected_slices = torch.tensor([[0, 8], [8, 40]])
        torch.testing.assert_close(decoder.slice_list, expected_slices)

        assert decoder.h.shape == (1, 40)
        assert decoder.c.shape == (1, 40)

        assert decoder.lin.in_features == 32
        assert decoder.lin.out_features == 100

    def test_forward_batch_first(self):
        decoder = LSTMDecoder(output_dim=100, hidden_dim=8, num_layers=2, dropout=0.1, batch_first=True)
        x = torch.randn(2, 5, 8)
        out, hidden = decoder(x)
        assert out.shape == (2, 5, 100)
        assert hidden[0].shape == (1, 2, 40)
        assert hidden[1].shape == (1, 2, 40)

    def test_forward_batch_first_with_hidden(self):
        decoder = LSTMDecoder(output_dim=100, hidden_dim=8, num_layers=2, dropout=0.1, batch_first=True)
        hidden_in = (decoder.h.expand(2, -1).unsqueeze(dim=0), decoder.c.expand(2, -1).unsqueeze(dim=0))
        x = torch.randn(2, 5, 8)
        out, hidden = decoder(x, hidden_in)
        assert out.shape == (2, 5, 100)
        assert hidden[0].shape == (1, 2, 40)
        assert hidden[1].shape == (1, 2, 40)


class TestLSTMAutoEncoder:
    def test_initialization(self):
        autoencoder = LSTMAutoEncoder(
            input_dim=100, hidden_dim=8, num_layers=2, batch_first=True, dropout=0.1, return_hidden=True
        )
        assert isinstance(autoencoder.encoder, LSTMEncoder)
        assert isinstance(autoencoder.decoder, LSTMDecoder)
        assert autoencoder.h.shape == (1, 80)
        assert autoencoder.c.shape == (1, 80)
        assert autoencoder.return_hidden is True

    def test_forward_return_hidden(self):
        autoencoder = LSTMAutoEncoder(
            input_dim=100, hidden_dim=8, num_layers=2, batch_first=True, dropout=0.1, return_hidden=True
        )
        x = torch.randn(2, 5, 100)
        out, hidden = autoencoder(x)
        assert out.shape == (2, 5, 100)
        assert hidden[0].shape == (1, 2, 80)
        assert hidden[1].shape == (1, 2, 80)

    def test_forward_not_return_hidden(self):
        autoencoder = LSTMAutoEncoder(
            input_dim=100, hidden_dim=8, num_layers=2, batch_first=True, dropout=0.1, return_hidden=False
        )
        x = torch.randn(2, 5, 100)
        out = autoencoder(x)
        assert out.shape == (2, 5, 100)

    def test_dtype_property(self):
        autoencoder = LSTMAutoEncoder(
            input_dim=100, hidden_dim=8, num_layers=2, batch_first=True, dropout=0.1, return_hidden=True
        )
        first_param_dtype = next(autoencoder.parameters()).dtype
        assert autoencoder.dtype == first_param_dtype
        autoencoder_double = autoencoder.double()
        assert autoencoder_double.dtype == torch.float64
