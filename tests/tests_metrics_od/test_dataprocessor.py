# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import polars as pl
import pytest

from pathlib import Path
from unittest.mock import MagicMock, patch

from adf.metrics.preprocessing.dataprocessor import MetricsDataProcessor
from adf.metrics.preprocessing.steps import MetricsNormalizer


@pytest.fixture
def processor():
    return MetricsDataProcessor()


@pytest.fixture
def input_df():
    return pl.DataFrame({"Time": ["2024-01-01", "2024-02-01", "2024-03-01"], "metric1": [1.0, 2.0, 3.0]}).with_columns(
        pl.col("Time").str.to_date()
    )


@pytest.fixture
def cutoff_dates():
    return {"train_end": "2024-01-31", "test_start": "2024-02-01"}


@pytest.fixture
def sample_dataframe():
    np.random.seed(42)
    return pl.DataFrame({"metric1": np.random.normal(10, 2, 100), "metric2": np.random.normal(20, 5, 100)})


def test_split_train_test(processor, tmp_path, input_df, cutoff_dates):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output"
    output_path.mkdir()

    with (
        patch("adf.metrics.preprocessing.dataprocessor.load_dataframe", return_value=input_df) as mock_load,
        patch("adf.metrics.preprocessing.dataprocessor.save_dataframe") as mock_save,
    ):
        train_files, val_files, test_files = processor.split_train_test(
            input_paths=[input_path], output_path=output_path, cutoff_dates=cutoff_dates, remove_local=False
        )

        assert len(train_files) == 1
        assert len(val_files) == 0
        assert len(test_files) == 1
        mock_load.assert_called_once_with(input_path, fmt="parquet", date_columns=["Time"])
        assert mock_save.call_count == 2


def test_split_train_test_invalid_dates(processor):
    with pytest.raises(ValueError):
        processor.split_train_test(
            input_paths=[], output_path=Path("."), cutoff_dates={"train_end": "2024-02-01", "test_start": "2024-01-31"}
        )


def test_process_training_data(processor, tmp_path, input_df):
    input_path = tmp_path / "train.parquet"
    output_path = tmp_path / "processed"
    scalers_path = tmp_path / "scalers.json"
    output_path.mkdir()

    mock_normalizer = MagicMock(spec=MetricsNormalizer)
    with (
        patch("adf.metrics.preprocessing.dataprocessor.load_dataframe", return_value=input_df),
        patch("adf.metrics.preprocessing.dataprocessor.save_dataframe") as mock_save,
        patch("adf.metrics.preprocessing.dataprocessor.prepare_metrics_data") as mock_prepare,
        patch.object(mock_normalizer, "save_scalers") as mock_save_scalers,
    ):
        mock_prepare.return_value = (input_df, mock_normalizer)

        n_rows = processor.process_training_data(
            input_paths=[input_path],
            output_path=output_path,
            scalers_path=scalers_path,
            metric_columns=["metric1"],
            anomaly_configs=[],
            remove_local=False,
        )

        mock_save.assert_called_once_with(input_df, output_path / input_path.name)

        assert n_rows == input_df.height

        mock_prepare.assert_called_once()
        mock_save_scalers.assert_called_once_with(scalers_path)


def test_process_inference_data(processor, tmp_path, input_df):
    input_path = tmp_path / "test.parquet"
    output_path = tmp_path / "inference"
    scalers_path = tmp_path / "scalers.json"
    output_path.mkdir()

    mock_normalizer = MagicMock(spec=MetricsNormalizer)
    with (
        patch("adf.metrics.preprocessing.dataprocessor.load_dataframe", return_value=input_df),
        patch("adf.metrics.preprocessing.dataprocessor.save_dataframe") as mock_save,
        patch(
            "adf.metrics.preprocessing.steps.MetricsNormalizer.load_scalers", return_value=mock_normalizer
        ) as mock_load_scalers,
        patch("adf.metrics.preprocessing.dataprocessor.prepare_metrics_data") as mock_prepare,
    ):
        mock_prepare.return_value = (input_df, None)

        n_rows = processor.process_inference_data(
            input_paths=[input_path],
            output_path=output_path,
            scalers_path=scalers_path,
            metric_columns=["metric1"],
            remove_local=False,
        )

        mock_save.assert_called_once_with(input_df, output_path / input_path.name)

        assert n_rows == input_df.height

        mock_prepare.assert_called_once()
        mock_load_scalers.assert_called_once_with(scalers_path)


def test_save_parameters(processor, tmp_path):
    source_path = tmp_path / "params.json"

    processor.save_parameters(
        total_num=100,
        source_path=source_path,
        metric_columns=["metric1"],
        time_units=["day"],
        cutoff_dates={"train_end": "2024-01-31", "test_start": "2024-02-01"},
        anomaly_configs=[],
    )

    assert source_path.exists()

    with open(source_path) as f:
        params = f.read()
        assert "metric1" in params
        assert "2024-01-31" in params


def test_metrics_normalizer_fit_transform(sample_dataframe):
    normalizer = MetricsNormalizer()

    df_transformed = normalizer.fit_transform(sample_dataframe, metric_columns=["metric1", "metric2"])

    assert df_transformed.shape == sample_dataframe.shape
    assert normalizer.scalers is not None
    assert "metric1" in normalizer.scalers
    assert "metric2" in normalizer.scalers

    for col in ["metric1", "metric2"]:
        mean = df_transformed[col].mean()
        std = df_transformed[col].std()
        assert np.isclose(mean, 0, atol=1e-6)
        assert np.isclose(std, 1, atol=1e-2)


def test_metrics_normalizer_transform(sample_dataframe):
    normalizer = MetricsNormalizer()
    normalizer.fit_transform(sample_dataframe, metric_columns=["metric1", "metric2"])

    new_data = pl.DataFrame({"metric1": np.random.normal(15, 3, 50), "metric2": np.random.normal(25, 4, 50)})

    transformed = normalizer.transform(new_data, metric_columns=["metric1", "metric2"])
    assert transformed.shape == new_data.shape


def test_metrics_normalizer_save_and_load(tmp_path, sample_dataframe):
    normalizer = MetricsNormalizer()
    normalizer.fit_transform(sample_dataframe, metric_columns=["metric1", "metric2"])

    scaler_path = tmp_path / "scalers.json"
    normalizer.save_scalers(scaler_path)

    assert scaler_path.exists()

    loaded_normalizer = MetricsNormalizer.load_scalers(scaler_path)

    original_transformed = normalizer.transform(sample_dataframe, ["metric1", "metric2"])
    loaded_transformed = loaded_normalizer.transform(sample_dataframe, ["metric1", "metric2"])

    for col in ["metric1", "metric2"]:
        assert np.allclose(original_transformed[col], loaded_transformed[col])


def test_metrics_normalizer_transform_without_fit(sample_dataframe):
    normalizer = MetricsNormalizer()
    with pytest.raises(ValueError):
        normalizer.transform(sample_dataframe, metric_columns=["metric1", "metric2"])
