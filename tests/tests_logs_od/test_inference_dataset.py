# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import polars as pl
import pytest

from datetime import date, timedelta

from adf.logs.inference.inference_dataset import ArrowRecordsDataset


@pytest.fixture
def sample_embeddings():
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int32)
    return embeddings, indices


@pytest.fixture
def sample_embeddings_empty():
    embeddings = np.array([], dtype=np.float32)
    indices = np.array([], dtype=np.int32)
    return embeddings, indices


@pytest.fixture
def sample_dataframe():
    return pl.DataFrame(
        {
            "logs": ["log1", "log2", "log3", "log4", "log5"],
            "timestamp": [date(2024, 1, 1) + timedelta(days=i) for i in range(5)],
        }
    )


@pytest.fixture
def sample_dataframe_int():
    return pl.DataFrame(
        {"logs": ["log1", "log2", "log3", "log4", "log5"], "timestamp": [i for i in range(5)]}
    ).with_columns(pl.col("timestamp").cast(pl.Int64))


@pytest.fixture
def sample_dataframe_small():
    return pl.DataFrame({"logs": ["log1", "log2"], "timestamp": [date(2024, 1, 1), date(2024, 1, 2)]})


def test_arrow_records_dataset_len(sample_dataframe_int):
    with pytest.raises(pl.exceptions.SchemaError):
        ArrowRecordsDataset(sample_dataframe_int, column="logs", datetime_column="timestamp", window_size=3)


def test_arrow_records_dataset_len_small(sample_dataframe_small):
    with pytest.raises(pl.exceptions.NoDataError):
        ArrowRecordsDataset(sample_dataframe_small, column="logs", datetime_column="timestamp", window_size=3)


def test_arrow_records_dataset_getitem_empty(sample_dataframe_int):
    with pytest.raises(pl.exceptions.SchemaError):
        dataset = ArrowRecordsDataset(sample_dataframe_int, column="logs", datetime_column="timestamp", window_size=3)
        dataset[0]


def test_arrow_records_dataloader_empty(sample_dataframe_int):
    with pytest.raises(pl.exceptions.SchemaError):
        dataset = ArrowRecordsDataset(sample_dataframe_int, column="logs", datetime_column="timestamp", window_size=3)
        list(dataset.as_dataloader(batch_size=2))


def test_arrow_records_dataset_window_size_too_large(sample_dataframe):
    large_window_size = 10
    with pytest.raises(pl.exceptions.NoDataError):
        ArrowRecordsDataset(sample_dataframe, column="logs", datetime_column="timestamp", window_size=large_window_size)
