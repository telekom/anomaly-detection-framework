# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import polars as pl

from datetime import datetime

from adf.metrics.preprocessing.steps import (
    MetricsNormalizer,
    compute_cyclic_features,
    filter_metrics_by_datetime,
    prepare_metrics_data,
)

DF = pl.DataFrame(
    {
        "Time": [
            datetime(2023, 1, 1, 0, 0, 0),
            datetime(2023, 1, 1, 12, 0, 0),
            datetime(2023, 1, 2, 6, 0, 0),
            datetime(2023, 1, 2, 18, 0, 0),
        ],
        "value": [10, 20, 30, 40],
    }
).with_columns(pl.col("Time").cast(pl.Datetime))


def test_filter_metrics_by_datetime_full_day():
    exclude_ranges = [{"date": "2023-01-01"}]
    df_filtered = filter_metrics_by_datetime(DF, "Time", exclude_ranges)
    times = df_filtered["Time"].to_list()
    assert all(dt.date() == datetime(2023, 1, 2).date() for dt in times)


def test_filter_metrics_by_datetime_time_range():
    exclude_ranges = [{"date": "2023-01-02", "start_time": "05:00:00", "end_time": "07:00:00"}]
    df_filtered = filter_metrics_by_datetime(DF, "Time", exclude_ranges)
    times = [
        dt.strftime("%H:%M:%S") for dt in df_filtered["Time"].to_list() if dt.date() == datetime(2023, 1, 2).date()
    ]
    assert "06:00:00" not in times


def test_compute_cyclic_features():
    df_input = pl.DataFrame({"Time": [datetime(2023, 1, 1, 0, 0, 0)]}).with_columns(pl.col("Time").cast(pl.Datetime))
    df_out = compute_cyclic_features(df_input, "Time", ["hour", "day"])
    hour_sin = df_out["hour_sin"][0]
    hour_cos = df_out["hour_cos"][0]
    day_sin = df_out["day_sin"][0]
    day_cos = df_out["day_cos"][0]
    np.testing.assert_allclose(hour_sin, 0, atol=5e-4)
    np.testing.assert_allclose(hour_cos, 1, atol=5e-4)
    np.testing.assert_allclose(day_sin, 0, atol=5e-4)
    np.testing.assert_allclose(day_cos, 1, atol=5e-4)
    df_no_week = compute_cyclic_features(df_input, "Time", ["hour", "unsupported"])
    assert "unsupported_sin" not in df_no_week.columns


def test_metrics_normalizer_fit_transform():
    normalizer = MetricsNormalizer()
    df_norm = normalizer.fit_transform(DF, ["value"])
    norm_vals = np.array(df_norm["value"].to_list(), dtype=float)
    np.testing.assert_allclose(norm_vals.mean(), 0, atol=1e-6)
    assert "value" in normalizer.scalers


def test_metrics_normalizer_transform():
    normalizer = MetricsNormalizer()
    normalizer.fit_transform(DF, ["value"])
    df_new = pl.DataFrame({"Time": [datetime(2023, 1, 3, 12, 0, 0)], "value": [50]}).with_columns(
        pl.col("Time").cast(pl.Datetime)
    )
    df_trans = normalizer.transform(df_new, ["value"])
    scaler = normalizer.scalers["value"]
    expected = scaler.transform(np.array([50]).reshape(-1, 1)).flatten()
    np.testing.assert_allclose(np.array(df_trans["value"].to_list()), expected)


def test_save_and_load_scalers(tmp_path):
    normalizer = MetricsNormalizer()
    normalizer.fit_transform(DF, ["value"])
    scaler_path = tmp_path / "scalers.pkl"
    normalizer.save_scalers(scaler_path)
    loaded_norm = MetricsNormalizer.load_scalers(scaler_path)
    assert set(loaded_norm.scalers.keys()) == {"value"}


def test_prepare_metrics_data_training():
    anomaly_configs = [{"date": "2023-01-01"}]
    df_input = pl.DataFrame(
        {"Time": [datetime(2023, 1, 1, 10, 0, 0), datetime(2023, 1, 2, 10, 0, 0)], "value": [100, 200]}
    ).with_columns(pl.col("Time").cast(pl.Datetime))
    df_proc, norm = prepare_metrics_data(df_input, "Time", ["value"], anomaly_configs, ["hour", "day"])
    assert df_proc.height == 1
    for col in ["hour_sin", "hour_cos", "day_sin", "day_cos"]:
        assert col in df_proc.columns
    assert "value" in df_proc.columns
    assert norm is not None


def test_prepare_metrics_data_inference():
    anomaly_configs = []
    norm_pre = MetricsNormalizer()
    norm_pre.fit_transform(DF, ["value"])
    df_proc, ret_norm = prepare_metrics_data(
        DF, "Time", ["value"], anomaly_configs, ["hour", "day"], normalizer=norm_pre
    )
    assert ret_norm is None
    for col in ["hour_sin", "hour_cos", "day_sin", "day_cos"]:
        assert col in df_proc.columns
    assert "value" in df_proc.columns
