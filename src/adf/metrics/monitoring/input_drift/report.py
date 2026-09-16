# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

"""JSON report and drift map plot."""

from __future__ import annotations

import json
import matplotlib

from datetime import date
from pathlib import Path
from typing import Any

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adf.metrics.monitoring.input_drift.constants import CLASS_COLOR_MAP
from adf.metrics.monitoring.input_drift.types import DriftReportDict, ThresholdsDict, Tier0CheckDict


def build_report(
    *,
    study_name: str,
    run_prefix: str,
    ref_start: date,
    ref_end: date,
    test_start: date,
    test_end: date,
    ref_rows: int,
    test_rows: int,
    tier0_status: str,
    tier0_checks: list[Tier0CheckDict],
    tier0_group_zero_df: pd.DataFrame,
    tier1_df: pd.DataFrame,
    aggregation: dict[str, Any],
    thresholds: ThresholdsDict,
    exclude_groups: list[str],
) -> DriftReportDict:
    """Build the JSON-serializable drift monitor report dict."""
    class_counts = tier1_df["input_drift_class"].value_counts().to_dict()
    group_rollup = (
        tier1_df.groupby("group")["input_drift_class"].value_counts().unstack(fill_value=0)
        if not tier1_df.empty
        else pd.DataFrame()
    )
    investigate_df = tier1_df[tier1_df["input_drift_class"] == "investigate"]

    return {
        "study": study_name,
        "run_prefix": run_prefix,
        "reference": {"start": str(ref_start), "end": str(ref_end), "rows": ref_rows},
        "test": {"start": str(test_start), "end": str(test_end), "rows": test_rows},
        "tier0_status": tier0_status,
        "tier0_checks": tier0_checks,
        "tier0_group_zero_rates": tier0_group_zero_df.to_dict(orient="records"),
        "tier1_class_counts": class_counts,
        "tier1_group_rollup": group_rollup.reset_index().to_dict(orient="records"),
        "tier1_all_features": tier1_df.to_dict(orient="records"),
        "tier1_investigate": investigate_df.to_dict(orient="records"),
        "aggregation": aggregation,
        "exclude_groups": exclude_groups,
        "methodology": {
            "input_only": True,
            "classification": "effect_size_first_psi_diagnostic_only",
            "thresholds": thresholds,
        },
    }


def write_report_json(report: DriftReportDict, path: Path) -> None:
    """Write drift report JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(report, f, indent=2, default=str)


def save_drift_map(tier1_df: pd.DataFrame, path: Path, thresholds: ThresholdsDict, title: str) -> None:
    """Save median-shift vs PSI scatter plot for non-sparse, non-excluded features."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = tier1_df[~tier1_df["sparse_metric"] & (tier1_df["input_drift_class"] != "excluded")].copy()
    if plot_df.empty:
        return

    plot_df["abs_median_shift"] = plot_df["median_shift_pct"].abs().replace([np.inf], np.nan)
    fig, ax = plt.subplots(figsize=(10, 6))
    for cls, sub in plot_df.groupby("input_drift_class"):
        ax.scatter(
            sub["abs_median_shift"],
            sub["psi"],
            label=cls,
            alpha=0.75,
            c=CLASS_COLOR_MAP.get(cls, "blue"),
            s=50,
            edgecolors="k",
            linewidths=0.3,
        )
    ax.axvline(float(thresholds["median_shift_investigate_pct"]), color="red", linestyle="--", alpha=0.5)
    ax.axhline(float(thresholds.get("psi_report_severe", 0.25)), color="red", linestyle=":", alpha=0.5)
    ax.set_xlabel("|median shift| % (daypart medians)")
    ax.set_ylabel("PSI (diagnostic)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
