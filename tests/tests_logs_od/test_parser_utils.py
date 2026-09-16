# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import polars as pl
import pytest
import re

from collections.abc import Callable

from adf.logs.parsers.drain.utils import _as_polars_dataframe, _check_clustering_metric, _check_X


class DummyConfig:
    def __init__(self, masking_instructions=None, mask_prefix="<", mask_suffix=">"):
        self.masking_instructions = masking_instructions or []
        self.mask_prefix = mask_prefix
        self.mask_suffix = mask_suffix


class DummyTemplateMiner:
    def __init__(self, config):
        self.config = config
        self.masker = None


def test_check_clustering_metric_valid():
    metric_func = _check_clustering_metric("calinski_harabasz_score")
    assert isinstance(metric_func, Callable)
    X = np.random.rand(10, 3)
    labels = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
    result = metric_func(X, labels)
    assert isinstance(result, (int, float, np.number))


def test_check_clustering_metric_invalid():
    with pytest.raises(ValueError):
        _check_clustering_metric("not_a_metric")


def dummy_load_dataframe(x, **kwargs):
    return pl.DataFrame({"col": ["value1", "value2"]})


def test_as_polars_dataframe_with_dataframe():
    df = pl.DataFrame({"a": [1, 2]})
    result = _as_polars_dataframe(df)
    assert result.equals(df)


def test_as_polars_dataframe_with_list():
    data = [{"a": 1}, {"a": 2}]
    result = _as_polars_dataframe(data)
    expected = pl.from_records(data)
    assert result.equals(expected)


def test_as_polars_dataframe_with_numpy():
    arr = np.array([[1, 2], [3, 4]])
    result = _as_polars_dataframe(arr)
    expected = pl.DataFrame(arr, schema=["column_0", "column_1"])
    assert result.equals(expected)


def test_as_polars_dataframe_invalid_type():
    with pytest.raises(TypeError):
        _as_polars_dataframe(123)


def test_check_X_valid(monkeypatch):
    df = pl.DataFrame({"col": ["a", "b", "a"]})
    monkeypatch.setattr("adf.core.dataframe.api.add_replaced_string", lambda df, column, pattern, replacement: df)
    result_unique = _check_X(df, return_unique=True)
    assert isinstance(result_unique, list)
    result_all = _check_X(df, return_unique=False)
    assert len(result_all) == df.shape[0]


def test_check_X_unique_preserves_input_order():
    """Deduplication must keep first-appearance order.

    Drain is order-dependent: the first message to reach a cluster creates it and later ones
    merge into it. An unordered unique() therefore makes fit() non-deterministic — the same
    corpus yields different templates and different cluster ids on every run.
    """
    values = [f"msg {i}" for i in range(500)]
    df = pl.DataFrame({"col": values + values})  # every message duplicated

    result = _check_X(df, return_unique=True)

    assert result == values, "unique() reordered the input"
    assert all(_check_X(df, return_unique=True) == result for _ in range(3)), "unstable across calls"


def test_check_X_invalid_columns():
    df = pl.DataFrame({"col1": ["a", "b"], "col2": ["c", "d"]})
    with pytest.raises(ValueError):
        _check_X(df)


def test_check_X_invalid_dtype():
    df = pl.DataFrame({"col": [1, 2, 3]})
    with pytest.raises(ValueError):
        _check_X(df)


# def test_check_X_with_nulls():
#     df = pl.DataFrame({"col": ["a", None, "b"]})
#     with pytest.raises(ValueError):
#         _check_X(df)
