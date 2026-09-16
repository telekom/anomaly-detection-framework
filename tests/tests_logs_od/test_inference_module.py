# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from unittest.mock import MagicMock

from adf.logs.inference.inference_module import DSAInferencePipeline
from adf.logs.models.autoencoder.model import LogAutoEncoder
from adf.logs.models.sentence_embedder.sentence_embedder import SimCSETransformer
from adf.logs.parsers.drain.drain import DrainLogParser


@pytest.fixture
def mock_dsa_inference():
    parser = MagicMock(spec=DrainLogParser)
    sentence_transformer = MagicMock(spec=SimCSETransformer)
    autoencoder = MagicMock(spec=LogAutoEncoder)

    parser.templates = [
        MagicMock(cluster_id=1, log_template_tokens=["template", "one"]),
        MagicMock(cluster_id=2, log_template_tokens=["template", "two"]),
    ]
    parser.cluster_ids = [1, 2]
    parser.rust_pattern = None
    parser.trim_max_length = None

    autoencoder.model_dtype = torch.float32
    sentence_transformer.encode = MagicMock(return_value=torch.rand(2, 384, device="cpu"))

    return DSAInferencePipeline(
        parser_model_or_path=parser,
        sentence_transformer_model_or_path=sentence_transformer,
        autoencoder_model_or_path=autoencoder,
        device_sentence_transformer="cpu",
        device_autoencoder="cpu",
        threshold=0.5,
        window_size=40,
        embed_size=384,
    )


def test_create_embeddings_cache(mock_dsa_inference):
    mock_dsa_inference.st.encode.return_value = torch.rand(2, 384, device="cpu")
    mock_dsa_inference.create_cache()

    assert len(mock_dsa_inference.cluster_id_to_tensor) == 2


def test_sync_cache(mock_dsa_inference):
    mock_dsa_inference.cluster_id_to_tensor = {1: torch.rand(1, 384), 2: torch.rand(1, 384)}

    new_embeddings = torch.rand(1, 384)
    mock_dsa_inference.sync_cache([3], new_embeddings)

    assert 3 in mock_dsa_inference.cluster_id_to_tensor
    tensor = mock_dsa_inference.cluster_id_to_tensor[3]
    assert tensor.shape == (1, 384)


def test_encode_with_cache(mock_dsa_inference):
    mock_dsa_inference.cluster_id_to_tensor = {1: torch.rand(1, 384)}
    mock_dsa_inference.device_sentence_transformer = "cpu"

    tensor = mock_dsa_inference.encode_from_templates("log entry one", 1)
    assert tensor.shape == (1, 384)


def test_encode_without_cache(mock_dsa_inference):
    mock_dsa_inference.cluster_id_to_tensor = {}
    mock_dsa_inference.st.encode.return_value = torch.rand(1, 384)
    mock_dsa_inference.window_size = 1

    tensor = mock_dsa_inference.encode_from_templates("new log entry", -1)
    assert tensor.shape == (1, 384)


def test_infer(mock_dsa_inference):
    mock_dsa_inference.dynamic_encode_from_templates = MagicMock(return_value=torch.rand(1, 40, 384))
    mock_dsa_inference.autoencoder = MagicMock()
    mock_dsa_inference.autoencoder.predict_single.return_value = torch.rand(41)

    result = mock_dsa_inference.infer(
        [["log entry one"] * 40, "log entry one"],
        [[1] * 40, 1],
    )

    assert result.shape == torch.Size([41])


def test_infer_with_return_class(mock_dsa_inference):
    mock_dsa_inference.encode_from_templates = MagicMock(return_value=torch.rand(40, 384))
    mock_dsa_inference.autoencoder = MagicMock()
    mock_dsa_inference.autoencoder.predict_single.return_value = torch.rand(1)

    templates = ["log entry one"] * 40
    template_ids = [1] * 40
    result = mock_dsa_inference.infer(
        templates, template_ids, return_class=True, dynamic_batching=False
    )

    assert result.shape == torch.Size([1])
    assert result.dtype == torch.int16
