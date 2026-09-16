# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from adf.logs.models.autoencoder.masking import softmax_masking


@pytest.fixture(params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def device(request):
    return torch.device(request.param)


def test_no_sampling_small_tensor(device):
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], device=device)
    expected = torch.tensor([False, False, False, False, True], device=device)
    result = softmax_masking(scores, 0.96)
    torch.testing.assert_close(result, expected)


def test_output_shape_dtype(device):
    scores = torch.full((10, 5), 0.5, device=device)
    result = softmax_masking(scores, 0.96)
    assert result.shape == scores.shape
    assert result.dtype == torch.bool


def test_threshold_tensor(device):
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], device=device)
    threshold_tensor = torch.tensor([0.96, 0.94, 0.9], device=device)
    expected = torch.tensor([False, False, False, False, True], device=device)
    result = softmax_masking(scores, threshold_tensor)
    torch.testing.assert_close(result, expected)


def test_hard_mask_inclusion(device):
    torch.manual_seed(42)
    scores = torch.linspace(0, 1, steps=101, device=device)
    result = softmax_masking(scores, 0.96)
    default_threshold = torch.tensor([0.96, 0.94, 0.9], device=device)
    quantiles = scores.quantile(default_threshold)
    hard_mask = scores > quantiles[0]
    for idx in torch.nonzero(hard_mask, as_tuple=False):
        assert result[tuple(idx)].item() is True


def test_determinism(device):
    torch.manual_seed(123)
    scores = torch.randn(20, device=device)
    result1 = softmax_masking(scores, 0.95)
    torch.manual_seed(123)
    result2 = softmax_masking(scores, 0.95)
    torch.testing.assert_close(result1, result2)
