# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch.nn as nn

from unittest.mock import patch

from adf.core.lightning.dataset import WindowEmbeddingLookup


def test_raises_error_for_invalid_backend():
    embeddings = np.random.rand(10, 5)
    window_indexes = np.array([0, 1, 2])
    with pytest.raises(NotImplementedError, match="Backend invalid_backend not implemented"):
        WindowEmbeddingLookup(embeddings, window_indexes, backend="invalid_backend")


def test_retrieves_correct_embedding_numpy():
    embeddings = np.random.rand(10, 5)
    window_indexes = np.array([[0, 1, 2]])
    lookup = WindowEmbeddingLookup(embeddings, window_indexes, backend="numpy")
    result = lookup[0]
    assert np.array_equal(result, embeddings[:3, :])


def test_retrieves_correct_embedding_torch():
    weights = torch.rand(10, 5)
    embeddings = nn.Embedding.from_pretrained(weights)
    window_indexes = torch.tensor([[0, 1, 2]])
    lookup = WindowEmbeddingLookup(embeddings, window_indexes, backend="torch")
    result = lookup[0]
    assert torch.equal(result, weights[:3, :])


def test_raises_error_for_invalid_window_indexes_type():
    embeddings = np.random.rand(10, 5)
    window_indexes = "invalid_type"
    with pytest.raises(NotImplementedError):
        WindowEmbeddingLookup(embeddings, window_indexes, backend="numpy")


def test_creates_instance_from_numpy_arrays():
    embeddings = np.random.rand(10, 5)
    window_indexes = np.array([0, 1, 2])
    lookup = WindowEmbeddingLookup.from_ndarray(embeddings, window_indexes)
    assert isinstance(lookup, WindowEmbeddingLookup)
    assert lookup.backend == "numpy"


def test_creates_instance_from_torch_tensors():
    embeddings = torch.rand(10, 5)
    window_indexes = torch.tensor([0, 1, 2])
    lookup = WindowEmbeddingLookup.from_torch(embeddings, window_indexes)
    assert isinstance(lookup, WindowEmbeddingLookup)
    assert lookup.backend == "torch"


def test_loads_numpy_data_correctly():
    with patch("numpy.load", side_effect=[np.random.rand(10, 5), np.array([0, 1, 2])]) as mock_load:
        lookup = WindowEmbeddingLookup.load("embeddings.npy", "indexes.npy", backend="numpy")
        assert isinstance(lookup, WindowEmbeddingLookup)
        assert lookup.backend == "numpy"
        mock_load.assert_any_call("embeddings.npy")
        mock_load.assert_any_call("indexes.npy")


def test_loads_torch_data_correctly():
    with patch("torch.load", side_effect=[torch.rand(10, 5), torch.tensor([0, 1, 2])]) as mock_load:
        lookup = WindowEmbeddingLookup.load("embeddings.pt", "indexes.pt", backend="torch")
        assert isinstance(lookup, WindowEmbeddingLookup)
        assert lookup.backend == "torch"
        mock_load.assert_any_call("embeddings.pt", weights_only=True)
        mock_load.assert_any_call("indexes.pt", weights_only=True)
