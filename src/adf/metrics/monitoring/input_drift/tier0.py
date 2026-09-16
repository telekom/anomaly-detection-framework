# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

"""Tier 0 pipeline health checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from adf.metrics.monitoring.input_drift.stats import rows_per_day, sampling_interval_seconds, zero_rate
from adf.metrics.monitoring.input_drift.types import ThresholdsDict, Tier0CheckDict


def run_tier0(
    df_ref: pl.DataFrame,
    df_test: pl.DataFrame,
    feature_cols: list[str],
    group_names: list[str],
    rollup_metric_names: list[str],
    timestamp_col: str,
    thresholds: ThresholdsDict,
) -> tuple[str, list[Tier0CheckDict], pd.DataFrame]:
    """Run pipeline health gates before Tier 1 drift classification."""
    tier0_checks: list[Tier0CheckDict] = []
    tier0_status = "ok"
    tier0_group_zero_df = pd.DataFrame()

    if len(df_ref) == 0 or len(df_test) == 0:
        return "fail", tier0_checks, tier0_group_zero_df

    ref_rpd = rows_per_day(df_ref, timestamp_col)
    test_rpd = rows_per_day(df_test, timestamp_col)
    ref_mean_rpd = float(ref_rpd.mean())
    test_mean_rpd = float(test_rpd.mean())
    row_drop_pct = 100.0 * (ref_mean_rpd - test_mean_rpd) / ref_mean_rpd if ref_mean_rpd else np.nan

    ref_interval = sampling_interval_seconds(df_ref, timestamp_col)
    test_interval = sampling_interval_seconds(df_test, timestamp_col)

    ref_zeros = {col: zero_rate(df_ref, col) for col in feature_cols}
    test_zeros = {col: zero_rate(df_test, col) for col in feature_cols}

    group_zero_rows = []
    group_zero_fail = False
    group_zero_warn = False
    for group in group_names:
        cols = [f"{metric}_{group}" for metric in rollup_metric_names if f"{metric}_{group}" in feature_cols]
        if not cols:
            continue
        ref_z = float(np.mean([ref_zeros[c] for c in cols]))
        test_z = float(np.mean([test_zeros[c] for c in cols]))
        delta_pp = test_z - ref_z
        level = "ok"
        if abs(delta_pp) > float(thresholds["tier0_db_zero_fail_pp"]):
            level, group_zero_fail = "fail", True
        elif abs(delta_pp) > float(thresholds["tier0_db_zero_warn_pp"]):
            level, group_zero_warn = "warn", True
        group_zero_rows.append(
            {
                "group": group,
                "zero_rate_ref": round(ref_z, 4),
                "zero_rate_test": round(test_z, 4),
                "delta_pp": round(delta_pp, 4),
                "level": level,
            }
        )
    tier0_group_zero_df = pd.DataFrame(group_zero_rows)

    low_days = test_rpd[test_rpd < float(thresholds["tier0_min_rows_per_day"])]

    checks = [
        ("ref_rows", len(df_ref), "info", True),
        ("test_rows", len(df_test), "info", True),
        ("ref_mean_rows_per_day", round(ref_mean_rpd, 1), "info", True),
        ("test_mean_rows_per_day", round(test_mean_rpd, 1), "info", True),
        (
            "row_count_drop_pct",
            round(row_drop_pct, 2),
            "fail" if row_drop_pct > float(thresholds["tier0_row_count_drop_pct"]) else "ok",
            row_drop_pct <= float(thresholds["tier0_row_count_drop_pct"]),
        ),
        ("sampling_interval_sec_ref", ref_interval, "info", True),
        (
            "sampling_interval_sec_test",
            test_interval,
            "warn" if abs(ref_interval - test_interval) > 1 else "ok",
            abs(ref_interval - test_interval) <= 1,
        ),
        ("test_days_below_min_rows", len(low_days), "warn" if len(low_days) > 0 else "ok", len(low_days) == 0),
    ]
    for name, value, level, passed in checks:
        tier0_checks.append({"check": name, "value": value, "level": level, "passed": passed})
        if not passed and level == "fail":
            tier0_status = "fail"
        elif not passed and level == "warn" and tier0_status == "ok":
            tier0_status = "warn"

    if group_zero_fail:
        tier0_status = "fail"
    elif group_zero_warn and tier0_status == "ok":
        tier0_status = "warn"

    return tier0_status, tier0_checks, tier0_group_zero_df
