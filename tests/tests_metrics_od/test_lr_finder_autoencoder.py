# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import torch

from torch.utils.data import DataLoader, TensorDataset

from adf.core.lightning.hpo import run_lr_finder
from adf.metrics.models.autoencoder.model import MetricsAutoEncoder


def test_lr_finder_with_metrics_autoencoder() -> None:
    torch.manual_seed(0)
    model = MetricsAutoEncoder(
        input_dim=8,
        hidden_dim=2,
        num_layers=2,
        dropout=0.0,
    )

    x = torch.randn(8, 3, 8)
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
