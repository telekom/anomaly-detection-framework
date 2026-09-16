# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa
"""Reconstruction error scores for metrics autoencoders, for Kafka/Elastic consumers.

Per-feature MSE averages squared error over time; pooled MSE averages over time and features.
Contribution percent per metric is ``mse_i / (sum_j mse_j + eps) * 100``.
"""

from __future__ import annotations

import torch

from datetime import UTC, datetime
from torch import Tensor
from typing import Any


def _coerce_to_datetime(ts: Any) -> datetime:
    if isinstance(ts, datetime):
        return ts
    # pandas.Timestamp, numpy.datetime64 via pandas
    if hasattr(ts, "to_pydatetime"):
        return ts.to_pydatetime()  # type: ignore[no-any-return]
    raise TypeError(f"Unsupported timestamp type for ranked MSE payload: {type(ts)!r}")


def pooled_and_per_feature_mse(target: Tensor, reconstruction: Tensor) -> tuple[Tensor, Tensor]:
    """Mean squared error per feature and pooled, per batch window.

    Args:
        target: Ground-truth tensor ``(batch, time_steps, n_features)`` (``x_metrics``).
        reconstruction: Model output ``(batch, time_steps, n_features)`` (``x_hat``).

    Returns:
        Tuple of:
            - ``per_feature_mse``: ``(batch, n_features)`` — mean over time of squared errors.
            - ``pooled_mse``: ``(batch,)`` — mean over time and features (same as
              ``torch.nn.functional.mse_loss(..., reduction='none').mean((1, 2))``).

    """
    if target.shape != reconstruction.shape:
        raise ValueError(f"target and reconstruction shapes must match; got {target.shape} vs {reconstruction.shape}")
    squared = (target - reconstruction) ** 2
    per_feature_mse = squared.mean(dim=1)
    pooled_mse = squared.mean(dim=(1, 2))
    return per_feature_mse, pooled_mse


def format_window_timestamp_iso_z(dt: datetime) -> str:
    """Format a window end time as ``YYYY-MM-DDTHH:MM:SS.mmmZ`` for Kafka/Elastic consumers."""
    if dt.tzinfo is None:
        dt_utc = dt.replace(tzinfo=UTC)
    else:
        dt_utc = dt.astimezone(UTC)
    # Milliseconds to match common Elastic / sample payloads
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt_utc.microsecond // 1000:03d}Z"


def ranked_reconstruction_mse_per_window(
    per_window_mse: Tensor,
    window_timestamps: list[datetime],
    metric_names: list[str],
    *,
    top_k: int = 10,
    eps: float = 1e-12,
) -> list[dict[str, Any]]:
    """Build per-window dicts with ``@timestamp`` and ``rank_*`` keys for Kafka/Elastic consumers.

    For each window, metrics are sorted by per-metric MSE (time-averaged) descending.
    Contribution percent is ``mse_i / sum(mse) * 100``. If the sum of MSEs is
    ~0, contributions are split evenly across metrics.

    Args:
        per_window_mse: Float tensor of shape ``(num_windows, n_metrics)``.
        window_timestamps: Length ``num_windows``, same order as tensor rows.
        metric_names: Length ``n_metrics``, column order aligned with ``per_window_mse``.
        top_k: Maximum number of ranks to include per window (e.g. 10).
        eps: Numerical stability for division.

    Returns:
        List of length ``num_windows``; each element is a flat dict with ``@timestamp`` and
        ``rank_j_feature_name`` / ``rank_j_mse`` / ``rank_j_contribution_percent`` for
        ``j = 1 .. min(top_k, n_metrics)``.

    """
    if per_window_mse.dim() != 2:
        raise ValueError(f"per_window_mse must be 2D (num_windows, n_metrics); got {per_window_mse.shape}")
    num_windows, n_metrics = per_window_mse.shape[0], per_window_mse.shape[1]
    if len(window_timestamps) != num_windows:
        raise ValueError(f"window_timestamps length {len(window_timestamps)} != num_windows {num_windows}")
    if len(metric_names) != n_metrics:
        raise ValueError(f"metric_names length {len(metric_names)} != n_metrics {n_metrics}")

    mse_cpu = per_window_mse.detach().float().cpu()
    k_max = max(0, min(int(top_k), n_metrics))
    out: list[dict[str, Any]] = []

    for w in range(num_windows):
        mse_row = mse_cpu[w].clone()
        total = float(mse_row.sum().item())
        if total < eps:
            contrib = torch.full_like(mse_row, 100.0 / max(n_metrics, 1))
        else:
            contrib = mse_row / (mse_row.sum() + eps) * 100.0

        order = torch.argsort(mse_row, descending=True)
        ts = window_timestamps[w]
        ts_dt = _coerce_to_datetime(ts)
        doc: dict[str, Any] = {"@timestamp": format_window_timestamp_iso_z(ts_dt)}

        for rank_idx in range(k_max):
            feat_i = int(order[rank_idx].item())
            j = rank_idx + 1
            doc[f"rank_{j}_feature_name"] = str(metric_names[feat_i])
            doc[f"rank_{j}_mse"] = float(mse_row[feat_i].item())
            doc[f"rank_{j}_contribution_percent"] = float(contrib[feat_i].item())

        out.append(doc)

    return out
