# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 effect-size drift classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from typing import Any

from adf.metrics.monitoring.input_drift.features import feature_group
from adf.metrics.monitoring.input_drift.stats import (
    adaptive_psi_bins,
    compute_psi_adaptive,
    ks_test,
    robust_cohens_d,
    transform_frozen,
    zero_rate,
)
from adf.metrics.monitoring.input_drift.types import FrozenScalerDict, PsiResult, ThresholdsDict


def classify_input_drift(row: dict[str, Any], thresholds: ThresholdsDict) -> str:
    """Effect-size-first classification. PSI is computed for reporting only."""
    zd = abs(float(row.get("zero_rate_delta") or 0))
    if row.get("sparse_metric"):
        return "data_quality" if zd >= float(thresholds["zero_rate_dq_flag_pp"]) else "sparse"
    if zd >= float(thresholds["zero_rate_dq_flag_pp"]):
        return "data_quality"

    med = abs(float(row.get("median_shift_pct") or 0))
    if not np.isfinite(med):
        med = 0.0
    ks = float(row.get("ks_stat") or 0.0)
    robust_d_val = row.get("robust_d")
    rd = abs(float(robust_d_val)) if isinstance(robust_d_val, (int, float)) and np.isfinite(robust_d_val) else 0.0
    dist_change = ks >= float(thresholds["ks_stat_investigate"]) or rd >= float(thresholds["robust_d_investigate"])
    level_change = med >= float(thresholds["median_shift_investigate_pct"])
    if level_change and dist_change:
        return "investigate"
    if level_change or ks >= float(thresholds["ks_stat_watch"]) or rd >= float(thresholds["robust_d_watch"]):
        return "watch"
    return "stable"


def build_tier1_from_aggregates(
    df_ref: pl.DataFrame,
    df_test: pl.DataFrame,
    ref_agg: pl.DataFrame,
    test_agg: pl.DataFrame,
    feature_cols: list[str],
    group_names: list[str],
    frozen_scalers: FrozenScalerDict,
    aggregation_label: str,
    thresholds: ThresholdsDict,
    tier0_status: str,
    exclude_groups: list[str] | None = None,
) -> pd.DataFrame:
    """Classify per-feature drift from daypart aggregate reference vs test periods."""
    exclude_groups = exclude_groups or []
    rows: list[dict[str, Any]] = []
    for col in feature_cols:
        group = feature_group(col, group_names)
        if group in exclude_groups:
            rows.append(
                {
                    "feature": col,
                    "group": group,
                    "aggregation": aggregation_label,
                    "excluded": True,
                    "sparse_metric": False,
                    "n_periods_ref": 0,
                    "n_periods_test": 0,
                    "zero_rate_ref": round(zero_rate(df_ref, col), 4),
                    "zero_rate_test": round(zero_rate(df_test, col), 4),
                    "zero_rate_delta": round(zero_rate(df_test, col) - zero_rate(df_ref, col), 4),
                    "median_ref": None,
                    "median_test": None,
                    "median_shift_pct": None,
                    "robust_d": None,
                    "psi": None,
                    "psi_bins": 0,
                    "psi_signal": "n/a",
                    "ks_stat": None,
                    "ks_pvalue": None,
                    "input_drift_class": "excluded",
                    "actionable": False,
                }
            )
            continue

        z_ref = zero_rate(df_ref, col)
        z_test = zero_rate(df_test, col)
        z_delta = z_test - z_ref

        ref_d = ref_agg[col].drop_nulls().to_numpy()
        test_d = test_agg[col].drop_nulls().to_numpy()
        n_ref, n_test = len(ref_d), len(test_d)
        sparse = z_ref >= float(thresholds["zero_rate_sparse_threshold"])

        med_ref = float(np.median(ref_d)) if n_ref else np.nan
        med_test = float(np.median(test_d)) if n_test else np.nan
        iqr_ref = float(np.percentile(ref_d, 75) - np.percentile(ref_d, 25)) if n_ref else np.nan
        if med_ref == 0:
            med_shift_pct = np.nan if med_test == 0 else np.inf
        else:
            med_shift_pct = 100.0 * (med_test - med_ref) / med_ref
        rd = robust_cohens_d(med_ref, med_test, iqr_ref)

        ref_norm = transform_frozen(ref_d, frozen_scalers[col])
        test_norm = transform_frozen(test_d, frozen_scalers[col])
        psi_res: PsiResult
        if sparse or n_ref < int(thresholds["psi_min_ref_days"]):
            psi_res = {"psi": float("nan"), "signal": "n/a", "n_bins": 0}
        else:
            nbins = adaptive_psi_bins(n_ref, thresholds)
            psi_res = compute_psi_adaptive(ref_norm, test_norm, nbins, thresholds)

        ks_stat, ks_pval = ks_test(ref_norm, test_norm)

        psi_raw = psi_res["psi"]
        psi_out: float | Any = (
            round(float(psi_raw), 4) if isinstance(psi_raw, (int, float)) and np.isfinite(psi_raw) else psi_raw
        )

        row: dict[str, Any] = {
            "feature": col,
            "group": group,
            "aggregation": aggregation_label,
            "excluded": False,
            "sparse_metric": sparse,
            "n_periods_ref": n_ref,
            "n_periods_test": n_test,
            "zero_rate_ref": round(z_ref, 4),
            "zero_rate_test": round(z_test, 4),
            "zero_rate_delta": round(z_delta, 4),
            "median_ref": round(med_ref, 4) if np.isfinite(med_ref) else med_ref,
            "median_test": round(med_test, 4) if np.isfinite(med_test) else med_test,
            "median_shift_pct": round(med_shift_pct, 2) if np.isfinite(med_shift_pct) else med_shift_pct,
            "robust_d": round(rd, 3) if np.isfinite(rd) else rd,
            "psi": psi_out,
            "psi_bins": psi_res.get("n_bins", 0),
            "psi_signal": psi_res["signal"],
            "ks_stat": round(ks_stat, 4) if np.isfinite(ks_stat) else ks_stat,
            "ks_pvalue": ks_pval,
        }
        row["input_drift_class"] = classify_input_drift(row, thresholds)
        row["actionable"] = tier0_status != "fail" and row["input_drift_class"] in ("watch", "investigate")
        rows.append(row)

    return pd.DataFrame(rows)


def group_rollup(tier1_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot Tier 1 class counts per group."""
    if tier1_df.empty:
        return pd.DataFrame()
    return tier1_df.groupby("group")["input_drift_class"].value_counts().unstack(fill_value=0)
