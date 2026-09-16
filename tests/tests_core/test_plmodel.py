# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from lightning.pytorch import LightningDataModule
from lightning.pytorch.trainer import Trainer
from torch.utils.data import DataLoader, Dataset
from unittest.mock import patch

from src.adf.core.lightning.plmodel import PLModel


class DummyDataset(Dataset):
    def __init__(self, size=10):
        self.data = torch.randn(size, 5)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


@pytest.fixture
def model():
    class DummyPLModel(PLModel):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(5, 2)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self.linear(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            return loss

        def train_dataloader(self):
            return DataLoader(DummyDataset(), batch_size=2)

        def configure_optimizers(self):
            return [torch.optim.Adam(self.parameters(), lr=0.001)]

    return DummyPLModel()


@pytest.fixture
def dummy_dataloader():
    dataset = DummyDataset()
    return DataLoader(dataset, batch_size=2)


class DummyDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset = DummyDataset()

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=2)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=2)


def test_configure_optimizers(model):
    optimizers = model.configure_optimizers()
    assert isinstance(optimizers, list)
    assert len(optimizers) > 0
    assert isinstance(optimizers[0], torch.optim.Optimizer)


def test_fit_with_dataloader(model, dummy_dataloader):
    dummy_trainer = Trainer()
    with patch("src.adf.core.lightning.plmodel.Trainer", return_value=dummy_trainer):
        with patch.object(dummy_trainer, "fit") as fit_mock:
            model.fit(train_dataloaders=dummy_dataloader, val_dataloaders=dummy_dataloader)
            fit_mock.assert_called_once_with(
                model=model, train_dataloaders=dummy_dataloader, val_dataloaders=dummy_dataloader, datamodule=None
            )


def test_fit_with_datamodule(model):
    datamodule = DummyDataModule()
    dummy_trainer = Trainer()
    with patch("src.adf.core.lightning.plmodel.Trainer", return_value=dummy_trainer):
        with patch.object(dummy_trainer, "fit") as fit_mock:
            model.fit(datamodule=datamodule)
            fit_mock.assert_called_once_with(
                model=model, train_dataloaders=None, val_dataloaders=None, datamodule=datamodule
            )


def test_test_method(model):
    datamodule = DummyDataModule()
    dummy_trainer = Trainer()
    model._trainer = dummy_trainer
    with patch.object(dummy_trainer, "test") as test_mock:
        model.test(datamodule=datamodule, ckpt_path="dummy.ckpt")
        test_mock.assert_called_once_with(model, datamodule=datamodule, ckpt_path="dummy.ckpt")
