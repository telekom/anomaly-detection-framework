# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import torch

from torch.utils.data import DataLoader, TensorDataset

from adf.core.lightning.hpo import run_lr_finder
from adf.logs.models.autoencoder.model import LogAutoEncoder


def test_lr_finder_with_log_autoencoder() -> None:
    torch.manual_seed(0)
    model = LogAutoEncoder(
        model_name="LSTMAutoEncoder",
        model_args={
            "input_dim": 4,
            "hidden_dim": 2,
            "num_layers": 2,
            "dropout": 0.0,
            "enc_dropout": 0.0,
            "dec_dropout": 0.0,
            "sigma": 0.0,
        },
        compile_model=False,
    )

    x = torch.randn(8, 3, 4)
    dataloader = DataLoader(TensorDataset(x), batch_size=2, shuffle=False)

    lr = run_lr_finder(
        model,
        dataloader,
        optimizer_name="AdamW",
        optimizer_config={"weight_decay": 0.0},
        start_lr=1e-6,
        end_lr=1e-3,
        num_steps=5,
    )

    assert 1e-6 <= lr <= 1e-3
