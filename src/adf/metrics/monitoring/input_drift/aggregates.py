# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

"""Timezone-aware daypart median aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from zoneinfo import ZoneInfo


def daypart_medians_cet(
    df: pl.DataFrame,
    feature_cols: list[str],
    timestamp_col: str,
    tz: ZoneInfo | str = "Europe/Berlin",
    work_start: int = 8,
    work_end: int = 18,
    weekdays_only: bool = False,
) -> pl.DataFrame:
    """Two rows per calendar day (night, working) with per-feature medians of raw samples."""
    if not isinstance(tz, ZoneInfo):
        tz = ZoneInfo(str(tz))

    pdf = df.select([timestamp_col] + feature_cols).to_pandas()
    pdf[timestamp_col] = pd.to_datetime(pdf[timestamp_col], utc=True).dt.tz_convert(tz)
    pdf["_date"] = pdf[timestamp_col].dt.date
    pdf["_hour"] = pdf[timestamp_col].dt.hour
    pdf["_weekday"] = pdf[timestamp_col].dt.dayofweek < 5
    if weekdays_only:
        pdf = pdf[pdf["_weekday"]].copy()

    pdf["_daypart"] = np.where((pdf["_hour"] >= work_start) & (pdf["_hour"] < work_end), "working", "night")
    agg = pdf.groupby(["_date", "_daypart"], sort=False)[feature_cols].median(numeric_only=True).reset_index()
    daypart_order = {"night": 0, "working": 1}
    agg["_sort"] = agg["_daypart"].map(daypart_order)
    agg = agg.sort_values(["_date", "_sort"]).drop(columns="_sort")
    agg["_weekday"] = pd.to_datetime(agg["_date"]).dt.dayofweek < 5
    agg["_period"] = agg["_date"].astype(str) + "_" + agg["_daypart"]
    agg[timestamp_col] = pd.to_datetime(agg["_date"].astype(str) + " 12:00:00", utc=True)
    return pl.from_pandas(agg[[timestamp_col, "_date", "_daypart", "_weekday", "_period"] + feature_cols])
