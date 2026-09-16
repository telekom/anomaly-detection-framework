# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for adf.metrics.monitoring.input_drift."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl

from adf.metrics.monitoring.input_drift.aggregates import daypart_medians_cet
from adf.metrics.monitoring.input_drift.constants import DEFAULT_THRESHOLDS
from adf.metrics.monitoring.input_drift.tier1 import classify_input_drift


def _fixture_timeseries_df() -> pl.DataFrame:
    rows = []
    start = date(2026, 5, 5)
    for day_offset in range(7):
        d = (start + timedelta(days=day_offset)).isoformat()
        for hour in [2, 10, 16]:
            for minute in (0, 5):
                rows.append(
                    {
                        "timestamp": datetime.fromisoformat(f"{d}T{hour:02d}:{minute:02d}:00+00:00"),
                        "metric_a_grp1": 100.0 if hour >= 8 else 10.0,
                    }
                )
    return pl.DataFrame(rows)


def test_daypart_medians_all_days_more_periods_than_weekdays():
    df = _fixture_timeseries_df()
    cols = ["metric_a_grp1"]
    all_days = daypart_medians_cet(df, cols, "timestamp", weekdays_only=False)
    weekdays = daypart_medians_cet(df, cols, "timestamp", weekdays_only=True)
    assert len(all_days) > len(weekdays)
    assert len(weekdays) < len(all_days)


def test_daypart_weekdays_excludes_weekend_dates():
    df = _fixture_timeseries_df()
    cols = ["metric_a_grp1"]
    weekdays = daypart_medians_cet(df, cols, "timestamp", weekdays_only=True)
    dates = weekdays["_date"].to_list()
    assert all(str(d) not in ("2026-05-09", "2026-05-10") for d in dates)
    assert "2026-05-11" in [str(d) for d in dates]


def test_classify_investigate_requires_level_and_distribution():
    thresholds = dict(DEFAULT_THRESHOLDS)
    row = {
        "sparse_metric": False,
        "zero_rate_delta": 0.0,
        "median_shift_pct": 30.0,
        "ks_stat": 0.1,
        "robust_d": 0.1,
    }
    assert classify_input_drift(row, thresholds) == "watch"

    row["ks_stat"] = 0.4
    assert classify_input_drift(row, thresholds) == "investigate"


def test_classify_sparse_and_data_quality():
    thresholds = dict(DEFAULT_THRESHOLDS)
    sparse = {"sparse_metric": True, "zero_rate_delta": 0.02}
    assert classify_input_drift(sparse, thresholds) == "sparse"
    dq = {"sparse_metric": False, "zero_rate_delta": 0.15, "median_shift_pct": 0}
    assert classify_input_drift(dq, thresholds) == "data_quality"
