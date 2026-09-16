# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import torch

from torch.utils.data import WeightedRandomSampler

from adf.metrics.models.autoencoder.datamodule import MetricsDataModule, create_weighted_sampler


def test_create_weighted_sampler():
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    quantile_threshold = 0.8
    sampler = create_weighted_sampler(scores, quantile_threshold)

    expected_threshold = torch.quantile(scores, quantile_threshold)
    expected_weights = scores.clone()
    expected_weights[expected_weights > expected_threshold] = 0

    expected_weights = expected_weights.to(sampler.weights.dtype)

    assert torch.allclose(sampler.weights, expected_weights)
    assert sampler.num_samples == len(scores)
    assert sampler.replacement is True


def test_metricsdatamodule_setup_fit():
    data = torch.randn(100, 10, 5)
    data_val = torch.randn(20, 10, 5)
    dm = MetricsDataModule(data, data_val, batch_size=16, num_workers=0, use_weighted_sampling=False)
    dm.setup("fit")

    assert hasattr(dm, "train_dataset")
    assert hasattr(dm, "val_dataset")
    assert len(dm.train_dataset) == 100
    assert len(dm.val_dataset) == 20


def test_metricsdatamodule_setup_test():
    data = torch.randn(50, 10, 5)
    dm = MetricsDataModule(data, data, batch_size=16, num_workers=0)
    dm.setup("test")

    assert hasattr(dm, "test_dataset")
    assert len(dm.test_dataset) == 50


def test_train_dataloader_batch_size():
    data = torch.randn(32, 10, 5)
    dm = MetricsDataModule(data, data, batch_size=8, num_workers=0)
    dm.setup("fit")
    train_loader = dm.train_dataloader()

    for batch in train_loader:
        assert batch[0].shape[0] <= 8
        break


def test_sampler_property_random():
    data = torch.randn(10, 3, 4)
    dm = MetricsDataModule(data, data, use_weighted_sampling=False)
    assert dm.sampler is None


def test_sampler_property_weighted():
    data = torch.randn(10, 3, 4)
    dm = MetricsDataModule(data, data, threshold_perc=0.8, use_weighted_sampling=True)
    scores = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float)
    dm.scores = scores
    sampler = dm.sampler
    assert isinstance(sampler, WeightedRandomSampler)

    expected_threshold = torch.quantile(scores, 0.8)
    expected_weights = scores.clone()
    expected_weights[expected_weights > expected_threshold] = 0
    expected_weights = expected_weights.to(sampler.weights.dtype)

    assert torch.allclose(sampler.weights, expected_weights)
    assert sampler.num_samples == len(scores)


def test_val_and_test_dataloader():
    data = torch.randn(20, 10, 5)
    data_val = torch.randn(8, 10, 5)
    dm = MetricsDataModule(data, data_val, batch_size=4, num_workers=0)

    dm.setup("fit")
    val_loader = dm.val_dataloader()
    val_data = dm.val_dataset.tensors[0]
    concatenated_val = torch.cat([batch[0] for batch in val_loader], dim=0)
    assert torch.allclose(concatenated_val, val_data)

    dm.setup("test")
    test_loader = dm.test_dataloader()
    test_data = dm.test_dataset.tensors[0]
    concatenated_test = torch.cat([batch[0] for batch in test_loader], dim=0)
    assert torch.allclose(concatenated_test, test_data)
