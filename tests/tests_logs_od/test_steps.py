# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import polars as pl
import pytest

from unittest.mock import patch

from adf.logs.preprocessing.steps import prepare


@pytest.fixture
def mock_drop_incident_rows():
    with patch("adf.core.dataframe.api.drop_incident_rows") as mock:
        yield mock


@pytest.fixture
def mock_generate_windows():
    with patch("adf.core.dataframe.api.generate_windows") as mock:
        yield mock


def test_raises_error_if_logs_column_missing(mock_drop_incident_rows):
    normal = pl.DataFrame({"index": [1, 2, 3]})
    with pytest.raises(ValueError, match="Column 'logs' not found in the dataframe schema"):
        prepare(normal=normal, incidents=None, datetime_column="datetime", sequence_length=5, logs_column="logs")


def raises_error_if_index_column_missing(mock_drop_incident_rows):
    normal = pl.DataFrame({"logs": ["log1", "log2"]})
    with pytest.raises(ValueError, match="Dataframe must contain an index column 'index'"):
        prepare(normal=normal, incidents=None, datetime_column="datetime", sequence_length=5, logs_column="logs")


def returns_none_if_dataframe_empty_after_dropping(mock_drop_incident_rows):
    mock_drop_incident_rows.return_value = pl.DataFrame()
    normal = pl.DataFrame({"index": [1, 2, 3], "logs": ["log1", "log2", "log3"]})
    result = prepare(normal=normal, incidents=None, datetime_column="datetime", sequence_length=5, logs_column="logs")
    assert result is None


def returns_indices_and_windows_array(mock_drop_incident_rows, mock_generate_windows):
    mock_drop_incident_rows.return_value = pl.DataFrame({"datetime": [1, 2, 3], "logs": ["log1", "log2", "log3"]})
    mock_generate_windows.return_value = ([0, 1], np.array([[1, 2], [2, 3]]))
    normal = pl.DataFrame({"index": [1, 2, 3], "logs": ["log1", "log2", "log3"]})
    result = prepare(normal=normal, incidents=None, datetime_column="datetime", sequence_length=5, logs_column="logs")
    assert result == ([0, 1], np.array([[1, 2], [2, 3]]))


def returns_indices_windows_and_uniques(mock_drop_incident_rows, mock_generate_windows):
    mock_drop_incident_rows.return_value = pl.DataFrame({"datetime": [1, 2, 3], "logs": ["log1", "log2", "log3"]})
    mock_generate_windows.return_value = ([0, 1], np.array([[1, 2], [2, 3]]))
    normal = pl.DataFrame({"index": [1, 2, 3], "logs": ["log1", "log2", "log3"]})
    result = prepare(
        normal=normal,
        incidents=None,
        datetime_column="datetime",
        sequence_length=5,
        logs_column="logs",
        return_uniques=True,
    )
    assert result == ([0, 1], np.array([[1, 2], [2, 3]]), np.array(["log1", "log2", "log3"]))


def raises_error_if_logs_column_missing_for_uniques(mock_drop_incident_rows, mock_generate_windows):
    mock_drop_incident_rows.return_value = pl.DataFrame({"datetime": [1, 2, 3]})
    mock_generate_windows.return_value = ([0, 1], np.array([[1, 2], [2, 3]]))
    normal = pl.DataFrame({"index": [1, 2, 3]})
    with pytest.raises(ValueError, match="logs_column must be specified to return unique values"):
        prepare(
            normal=normal,
            incidents=None,
            datetime_column="datetime",
            sequence_length=5,
            logs_column=None,
            return_uniques=True,
        )
