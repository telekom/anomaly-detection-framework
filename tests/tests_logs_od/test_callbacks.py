# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from unittest.mock import MagicMock, patch

from adf.core.lightning.callbacks import EMAUpdateCallback


@pytest.fixture
def mock_trainer():
    trainer = MagicMock()
    trainer.datamodule = MagicMock()
    trainer.current_epoch = 0
    trainer.logger = MagicMock()
    trainer.max_epochs = 10
    return trainer


@pytest.fixture
def mock_pl_module():
    mock_module = MagicMock()
    mock_module.parameters = MagicMock(return_value=[])
    mock_module.validation_metrics = MagicMock()
    mock_module.test_metrics = MagicMock()
    datamodule = MagicMock()
    datamodule.scores = torch.tensor([0.1, 0.2])
    datamodule.threshold = 0.0
    datamodule.val_dataset = MagicMock()
    datamodule.batch_size = 32
    datamodule.backup_files = {"data": "dummy_data_data.npy"}
    datamodule.num_logs_memmap = 0
    datamodule.num_windows_memmap = 0
    datamodule.window_size = 10
    datamodule.input_dim = 5
    mock_module.datamodule = datamodule
    return mock_module


def test_ema_update_initialization():
    callback = EMAUpdateCallback(average_coef=0.99)
    assert callback.ema is None


def test_ema_update_on_train_epoch_end_initialization(mock_trainer, mock_pl_module):
    callback = EMAUpdateCallback(average_coef=0.99)
    with patch("adf.core.lightning.callbacks.ExponentialMovingAverage") as mock_ema_cls:
        mock_ema_instance = MagicMock()
        mock_ema_cls.return_value = mock_ema_instance
        callback.on_train_epoch_end(mock_trainer, mock_pl_module)
        mock_ema_cls.assert_called_once_with(mock_pl_module.parameters(), decay=0.99)
        assert callback.ema is mock_ema_instance


def test_ema_update_on_train_epoch_end_update(mock_trainer, mock_pl_module):
    callback = EMAUpdateCallback(average_coef=0.99)
    callback.ema = MagicMock()
    callback.on_train_epoch_end(mock_trainer, mock_pl_module)
    callback.ema.update.assert_called_once()
    callback.ema.copy_to.assert_called_once()


def test_ema_update_on_train_end(mock_trainer, mock_pl_module):
    callback = EMAUpdateCallback(average_coef=0.99)
    callback.ema = MagicMock()
    callback.on_train_end(mock_trainer, mock_pl_module)
    assert callback.ema is None
