# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

"""Default Tier 0 / Tier 1 thresholds for input drift monitoring."""

from adf.metrics.monitoring.input_drift.types import ThresholdsDict

DEFAULT_THRESHOLDS: ThresholdsDict = {
    "tier0_row_count_drop_pct": 20.0,
    "tier0_db_zero_warn_pp": 0.05,
    "tier0_db_zero_fail_pp": 0.15,
    "tier0_min_rows_per_day": 200,
    "psi_bins_max": 5,
    "psi_min_samples_per_bin": 5,
    "psi_min_ref_days": 14,
    "psi_report_severe": 0.25,
    "psi_report_moderate": 0.10,
    "median_shift_investigate_pct": 25.0,
    "median_shift_watch_pct": 15.0,
    "ks_stat_investigate": 0.35,
    "ks_stat_watch": 0.20,
    "robust_d_investigate": 0.70,
    "robust_d_watch": 0.40,
    "zero_rate_sparse_threshold": 0.90,
    "zero_rate_dq_flag_pp": 0.10,
}

SCALER_QUANTILE_RANGE = (5.0, 95.0)
IQR_EPSILON = 1e-6

CLASS_COLOR_MAP = {
    "stable": "green",
    "watch": "orange",
    "investigate": "red",
    "data_quality": "gray",
    "sparse": "lightgray",
    "excluded": "silver",
}

VALID_AGGREGATION_MODES = ("daypart_all_days", "daypart_weekdays")
