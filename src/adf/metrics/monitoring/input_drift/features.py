# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

"""Feature column helpers and date slicing."""

from __future__ import annotations

import polars as pl

from datetime import date


def build_grouped_feature_columns(metric_names: list[str], group_names: list[str]) -> list[str]:
    """Build wide-format column names ``{metric}_{group}`` for each metric × group pair."""
    return [f"{metric}_{group}" for group in group_names for metric in metric_names]


def feature_group(feature: str, group_names: list[str]) -> str:
    """Return the group suffix parsed from a wide-format feature column name."""
    for group in group_names:
        if feature.endswith(f"_{group}"):
            return group
    return "unknown"


def split_by_dates(df: pl.DataFrame, start: date, end: date, timestamp_col: str) -> pl.DataFrame:
    """Filter rows whose timestamp falls within ``start`` and ``end`` (inclusive)."""
    dtype = df.schema[timestamp_col]
    if dtype == pl.Utf8:
        ts_date = pl.col(timestamp_col).str.slice(0, 10).str.to_date()
    elif dtype == pl.Date:
        ts_date = pl.col(timestamp_col)
    else:
        ts_date = pl.col(timestamp_col).cast(pl.Date)
    return df.filter(ts_date.is_between(start, end))
