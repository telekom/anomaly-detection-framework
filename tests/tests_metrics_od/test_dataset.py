# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import torch

from adf.metrics.inference.inference_dataset import MetricsInferenceDataset


def test_dtype_conversion():
    data_int = torch.tensor([[1, 2, 3], [4, 5, 6]])
    dataset = MetricsInferenceDataset(data_int, tensor_dtype=torch.float32)
    assert dataset.data.dtype == torch.float32


def test_len():
    data = torch.randn(10, 5, 3)
    dataset = MetricsInferenceDataset(data)
    assert len(dataset) == 10


def test_getitem():
    data = torch.arange(24).reshape(4, 2, 3)
    dataset = MetricsInferenceDataset(data)
    for idx in range(len(dataset)):
        item = dataset[idx]
        assert torch.equal(item, data[idx])


def test_as_dataloader():
    data = torch.randn(10, 5, 3)
    dataset = MetricsInferenceDataset(data)
    batch_size = 4
    dataloader = dataset.as_dataloader(batch_size=batch_size, shuffle=False)

    total_items = 0
    for batch in dataloader:
        total_items += batch.shape[0]
        assert batch.shape[1] == 5
        assert batch.shape[2] == 3

    assert total_items == len(dataset)
