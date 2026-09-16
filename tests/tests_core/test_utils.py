# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import random
import torch

from src.adf.core.common.utils import assert_memmap_tensor_dtype, get_device_accelerator, make_deterministic


def test_make_deterministic():
    make_deterministic(123)

    # Check Python's random module
    random_value_1 = random.randint(0, 100)
    make_deterministic(123)
    random_value_2 = random.randint(0, 100)

    assert random_value_1 == random_value_2  # Ensures deterministic behavior


def test_assert_memmap_tensor_dtype_mismatch():
    with pytest.raises(
        ValueError, match="Tensor dtype torch.float64 does not match memmap dtype <class 'numpy.float32'>"
    ):
        assert_memmap_tensor_dtype(np.float32, torch.float64)


def test_assert_memmap_tensor_dtype_unsupported():
    with pytest.raises(ValueError, match="Unsupported dtype <class 'numpy.object_'>"):
        assert_memmap_tensor_dtype(np.object_, torch.float32)


def test_get_device_name_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert get_device_accelerator() == torch.device("cpu")


def test_get_device_name_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device_accelerator(device=0) == torch.device("cuda:0")


def test_get_device_name_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_device_accelerator() == torch.device("mps")


# Fix for the doctest in assert_memmap_tensor_dtype
def test_doctest_assert_memmap_tensor_dtype():
    from pytest import raises

    with raises(ValueError, match="Tensor dtype torch.float64 does not match memmap dtype <class 'numpy.float32'>"):
        assert_memmap_tensor_dtype(np.float32, torch.float64)
