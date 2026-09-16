# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

"""Drift statistics: PSI, KS, frozen scalers, zero rates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from collections import Counter
from scipy.stats import ks_2samp
from sklearn.preprocessing import RobustScaler
from typing import Any, cast

from adf.metrics.monitoring.input_drift.constants import IQR_EPSILON, SCALER_QUANTILE_RANGE
from adf.metrics.monitoring.input_drift.types import FrozenScalerDict, PsiResult, ThresholdsDict


def rows_per_day(df: pl.DataFrame, timestamp_col: str) -> pd.Series:
    """Count rows per calendar day from the timestamp column."""
    pdf = df.select(timestamp_col).to_pandas()
    pdf[timestamp_col] = pd.to_datetime(pdf[timestamp_col], utc=True)
    return pdf.set_index(timestamp_col).resample("D").size()


def sampling_interval_seconds(df: pl.DataFrame, timestamp_col: str, max_rows: int = 2000) -> float:
    """Return the most common sampling interval in seconds (from consecutive timestamps)."""
    pdf = df.select(timestamp_col).head(max_rows).to_pandas()
    pdf[timestamp_col] = pd.to_datetime(pdf[timestamp_col], utc=True)
    ts = pdf.sort_values(timestamp_col)[timestamp_col].tolist()
    if len(ts) < 2:
        return float("nan")
    diffs = [(ts[i + 1] - ts[i]).total_seconds() for i in range(len(ts) - 1)]
    return float(Counter(round(d, 1) for d in diffs).most_common(1)[0][0])


def zero_rate(df: pl.DataFrame, col: str) -> float:
    """Fraction of non-null values equal to zero."""
    vals = df[col].drop_nulls().to_numpy()
    return float(np.mean(vals == 0)) if len(vals) else float("nan")


def adaptive_psi_bins(n_ref: int, cfg: ThresholdsDict) -> int:
    """Choose PSI bin count from reference sample size and config."""
    per_bin = int(cfg["psi_min_samples_per_bin"])
    return max(2, min(int(cfg["psi_bins_max"]), n_ref // per_bin))


def compute_psi_adaptive(
    reference: np.ndarray[Any, Any], current: np.ndarray[Any, Any], n_bins: int, cfg: ThresholdsDict, eps: float = 1e-6
) -> PsiResult:
    """Compute adaptive-bin PSI between reference and current normalized samples."""
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < 2 or len(current) < 2 or n_bins < 2:
        return {"psi": np.nan, "signal": "n/a", "n_bins": n_bins}

    quantile_points = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(reference, quantile_points))
    if len(bin_edges) < 2:
        return {"psi": 0.0, "signal": "none", "n_bins": len(bin_edges) - 1}

    bin_edges[0] = min(bin_edges[0], current.min()) - 1e-9
    bin_edges[-1] = max(bin_edges[-1], current.max()) + 1e-9
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), eps, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), eps, None)
    psi = float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())
    severe = float(cfg.get("psi_report_severe", 0.25))
    moderate = float(cfg.get("psi_report_moderate", 0.10))
    signal = "severe" if psi >= severe else "moderate" if psi >= moderate else "none"
    return {"psi": psi, "signal": signal, "n_bins": len(bin_edges) - 1}


def robust_cohens_d(med_ref: float, med_test: float, iqr_ref: float) -> float:
    """Robust Cohen's d using reference IQR as scale."""
    if not np.isfinite(iqr_ref) or iqr_ref < IQR_EPSILON:
        return np.nan
    return float((med_test - med_ref) / iqr_ref)


def fit_frozen_scalers(df_ref: pl.DataFrame, feature_cols: list[str]) -> FrozenScalerDict:
    """Fit arcsinh + RobustScaler on reference data (frozen for test comparison)."""
    scalers: FrozenScalerDict = {}
    for col in feature_cols:
        vals = np.arcsinh(df_ref[col].drop_nulls().to_numpy().astype(np.float64))
        if len(vals) < 2:
            scalers[col] = None
            continue
        q25, q75 = np.percentile(vals, [25, 75])
        if (q75 - q25) < IQR_EPSILON:
            scalers[col] = None
            continue
        sc = RobustScaler(quantile_range=SCALER_QUANTILE_RANGE)
        sc.fit(vals.reshape(-1, 1))
        scalers[col] = sc
    return scalers


def transform_frozen(values: np.ndarray[Any, Any], scaler: RobustScaler | None) -> np.ndarray[Any, Any]:
    """Apply frozen scaler to arcsinh-transformed values."""
    if scaler is None:
        return np.zeros_like(values, dtype=float)
    v = np.arcsinh(values.astype(np.float64))
    return cast(np.ndarray[Any, Any], scaler.transform(v.reshape(-1, 1)).flatten())


def ks_test(ref_norm: np.ndarray[Any, Any], test_norm: np.ndarray[Any, Any]) -> tuple[float, float]:
    """Two-sample KS test on normalized reference and test arrays."""
    if len(ref_norm) >= 5 and len(test_norm) >= 5:
        ks_stat, ks_pval = ks_2samp(ref_norm, test_norm)
        return float(ks_stat), float(ks_pval)
    return np.nan, np.nan
