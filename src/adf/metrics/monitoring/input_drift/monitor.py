# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end input drift monitoring pipeline."""

from __future__ import annotations

import pandas as pd
import polars as pl

from datetime import date
from zoneinfo import ZoneInfo

from adf.metrics.monitoring.input_drift.aggregates import daypart_medians_cet
from adf.metrics.monitoring.input_drift.constants import DEFAULT_THRESHOLDS, VALID_AGGREGATION_MODES
from adf.metrics.monitoring.input_drift.features import split_by_dates
from adf.metrics.monitoring.input_drift.report import build_report
from adf.metrics.monitoring.input_drift.stats import fit_frozen_scalers
from adf.metrics.monitoring.input_drift.tier0 import run_tier0
from adf.metrics.monitoring.input_drift.tier1 import build_tier1_from_aggregates
from adf.metrics.monitoring.input_drift.types import DriftReportDict, ThresholdsDict


def run_input_drift_monitor(
    df: pl.DataFrame,
    feature_cols: list[str],
    group_names: list[str],
    rollup_metric_names: list[str],
    timestamp_col: str,
    ref_start: date,
    ref_end: date,
    test_start: date,
    test_end: date,
    thresholds: ThresholdsDict | None = None,
    exclude_groups: list[str] | None = None,
    aggregation_mode: str = "daypart_weekdays",
    timezone: str = "Europe/Berlin",
    work_start_hour: int = 8,
    work_end_hour: int = 18,
    study_name: str = "input_drift_monitor",
    run_prefix: str = "",
) -> tuple[DriftReportDict, pd.DataFrame]:
    """Run Tier 0 + Tier 1 input drift on reference vs test windows.

    Returns the JSON-serializable report dict and the Tier 1 feature-level DataFrame.
    """
    if aggregation_mode not in VALID_AGGREGATION_MODES:
        raise ValueError(f"aggregation_mode must be one of {VALID_AGGREGATION_MODES}, got {aggregation_mode!r}")

    resolved_thresholds: ThresholdsDict = dict(DEFAULT_THRESHOLDS) if thresholds is None else dict(thresholds)
    exclude_groups = list(exclude_groups or [])

    df_ref = split_by_dates(df, ref_start, ref_end, timestamp_col)
    df_test = split_by_dates(df, test_start, test_end, timestamp_col)

    tier0_status, tier0_checks, tier0_group_zero_df = run_tier0(
        df_ref, df_test, feature_cols, group_names, rollup_metric_names, timestamp_col, resolved_thresholds
    )

    weekdays_only = aggregation_mode == "daypart_weekdays"
    tz = ZoneInfo(timezone)

    ref_agg = daypart_medians_cet(
        df_ref,
        feature_cols,
        timestamp_col,
        tz=tz,
        work_start=work_start_hour,
        work_end=work_end_hour,
        weekdays_only=weekdays_only,
    )
    test_agg = daypart_medians_cet(
        df_test,
        feature_cols,
        timestamp_col,
        tz=tz,
        work_start=work_start_hour,
        work_end=work_end_hour,
        weekdays_only=weekdays_only,
    )

    frozen_scalers = fit_frozen_scalers(df_ref, feature_cols)
    tier1_df = build_tier1_from_aggregates(
        df_ref,
        df_test,
        ref_agg,
        test_agg,
        feature_cols,
        group_names,
        frozen_scalers,
        aggregation_label=aggregation_mode,
        thresholds=resolved_thresholds,
        tier0_status=tier0_status,
        exclude_groups=exclude_groups,
    )

    aggregation_meta = {
        "mode": aggregation_mode,
        "timezone": str(tz),
        "work_start_hour_local": work_start_hour,
        "work_end_hour_local": work_end_hour,
        "weekdays_only": weekdays_only,
        "n_periods_ref": len(ref_agg),
        "n_periods_test": len(test_agg),
    }

    report = build_report(
        study_name=study_name,
        run_prefix=run_prefix,
        ref_start=ref_start,
        ref_end=ref_end,
        test_start=test_start,
        test_end=test_end,
        ref_rows=len(df_ref),
        test_rows=len(df_test),
        tier0_status=tier0_status,
        tier0_checks=tier0_checks,
        tier0_group_zero_df=tier0_group_zero_df,
        tier1_df=tier1_df,
        aggregation=aggregation_meta,
        thresholds=resolved_thresholds,
        exclude_groups=exclude_groups,
    )
    return report, tier1_df
