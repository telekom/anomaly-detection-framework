# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import joblib
import logging
import numpy as np
import polars as pl

from datetime import datetime
from numpy.typing import NDArray
from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler
from typing import Any

from adf.core.common.logger import timelog as tl
from adf.core.dataframe import api as api


def filter_metrics_by_datetime(
    df: pl.DataFrame, timestamp_col: str, exclude_ranges: list[dict[str, Any]]
) -> pl.DataFrame:
    """Filter out metrics data points that fall within specified datetime ranges.

    Args:
        df: DataFrame containing metrics data
        timestamp_col: Name of column containing timestamps
        exclude_ranges: List of ranges to exclude, each with:
            - date (str): The date in YYYY-MM-DD format
            - start_time (str, optional): Start time HH:MM:SS
            - end_time (str, optional): End time HH:MM:SS
            If start_time/end_time not provided, excludes entire day

    Returns:
        Filtered DataFrame with anomalous periods removed

    """
    df_filtered = df.clone()

    for range_config in exclude_ranges:
        date = range_config["date"]

        # Full day exclusion
        if "start_time" not in range_config:
            exclude_start = datetime.strptime(f"{date} 00:00:00", "%Y-%m-%d %H:%M:%S")
            exclude_end = datetime.strptime(f"{date} 23:59:59", "%Y-%m-%d %H:%M:%S")
        else:
            # Time range exclusion
            exclude_start = datetime.strptime(f"{date} {range_config['start_time']}", "%Y-%m-%d %H:%M:%S")
            exclude_end = datetime.strptime(f"{date} {range_config['end_time']}", "%Y-%m-%d %H:%M:%S")

        # Keep only points outside the exclusion range
        df_filtered = df_filtered.filter(
            (pl.col(timestamp_col) < exclude_start) | (pl.col(timestamp_col) > exclude_end)
        )

    return df_filtered


@tl
def compute_cyclic_features(df: pl.DataFrame, timestamp_col: str, time_units: list[str] | None = None) -> pl.DataFrame:
    """Add cyclic time features using sin/cos transformations."""
    if time_units is None:
        time_units = ["hour", "day", "week"]
    timestamp_s = pl.col(timestamp_col).dt.timestamp()

    periods = {"hour": 60 * 60, "day": 24 * 60 * 60, "week": 7 * 24 * 60 * 60}
    new_columns = []

    for unit in time_units:
        if unit not in periods:
            logging.warning(f"Unsupported time unit: {unit}")
            continue

        period = periods[unit]
        angle = 2 * np.pi * timestamp_s / period

        # df = df.with_columns(df, colname=f"{unit}_sin", values=angle.sin().to_numpy())
        # df = df.with_columns(df, colname=f"{unit}_cos", values=angle.cos().to_numpy())

        new_columns.append((angle.sin()).alias(f"{unit}_sin"))
        new_columns.append((angle.cos()).alias(f"{unit}_cos"))
    df = df.with_columns(new_columns)
    return df


_SUPPORTED_TYPES = ("StandardScaler", "RobustScaler", "custom")


class MetricsNormalizer:
    """Handles normalization of metrics data with scaler persistence."""

    def __init__(self, normalization_type: str = "StandardScaler", scaler_params: dict[str, Any] | None = None) -> None:
        """Initialize normalizer.

        Args:
            normalization_type: "StandardScaler", "RobustScaler", or "custom".
                "custom" is an extension point: today it runs arcsinh + RobustScaler,
                but the pipeline is isolated in _custom_fit / _custom_transform so
                it can be replaced without touching anything else.
            scaler_params: Optional dict of parameters forwarded to the underlying
                scaler.  Recognised keys per type:
                    for RobustScaler ---> quantile_range: [lower, upper]
                    for custom ---> quantile_range: [lower, upper]
                                     iqr_epsilon:  float  (default 1e-6)
                Ignored for StandardScaler.

        """
        if normalization_type not in _SUPPORTED_TYPES:
            raise ValueError(f"Unsupported normalization_type '{normalization_type}'. Supported: {_SUPPORTED_TYPES}")
        self.normalization_type = normalization_type
        self.scaler_params: dict[str, Any] = scaler_params or {}
        self.scalers: dict[str, Any] = {}
        # columns zeroed out during fit (custom mode only)
        self._constant_columns: set[str] = set()

    def _build_scaler(self) -> StandardScaler | RobustScaler:
        """Return a fresh, unfitted scaler using self.scaler_params."""
        if self.normalization_type == "StandardScaler":
            return StandardScaler()

        # Both RobustScaler and custom use a RobustScaler underneath;
        # quantile_range comes from scaler_params with a safe default.
        quantile_range = tuple(self.scaler_params.get("quantile_range", [5.0, 95.0]))
        return RobustScaler(quantile_range=quantile_range)

    def _custom_fit(self, values: NDArray[np.float_]) -> tuple[NDArray[np.float_], bool]:
        """Fit-side pre-processing for the custom pipeline.

        Returns:
            (transformed values, is_constant)
                is_constant=True  then caller should store None and zero the column.

        """
        iqr_epsilon = self.scaler_params.get("iqr_epsilon", 1e-6)

        values_t = np.arcsinh(values)

        q25, q75 = np.percentile(values_t, [25, 75])
        if (q75 - q25) < iqr_epsilon:
            return values_t, True  # nearly -constant

        return values_t, False

    def _custom_transform(self, values: NDArray[np.float_]) -> NDArray[np.float_]:
        """Transform-side pre-processing for the custom pipeline."""
        return np.arcsinh(values).astype(np.float_)

    @tl
    def fit_transform(self, df: pl.DataFrame, metric_columns: list[str]) -> pl.DataFrame:
        """Fit scaler to data and transform.

        Args:
            df: Input DataFrame
            metric_columns: Columns to normalize

        Returns:
            DataFrame with normalized metrics

        """
        df_norm = df.clone()
        new_columns = []

        for col in metric_columns:
            if not api.isin_schema(df, column=col):
                logging.warning(f"Column {col} not found")
                continue

            values = df[col].to_numpy().astype(float)

            if self.normalization_type == "custom":
                values, is_constant = self._custom_fit(values)

                if is_constant:
                    self.scalers[col] = None
                    self._constant_columns.add(col)
                    new_columns.append(pl.Series(col, np.zeros(len(values))))
                    logging.warning(f"Column '{col}' is nearly constant, set to 0")
                    continue

            scaler = self._build_scaler()
            normalized = scaler.fit_transform(values.reshape(-1, 1))

            self.scalers[col] = scaler
            new_columns.append(pl.Series(col, normalized.flatten()))

        if new_columns:
            df_norm = df_norm.with_columns(new_columns)

        return df_norm

    @tl
    def transform(self, df: pl.DataFrame, metric_columns: list[str]) -> pl.DataFrame:
        """Transform data using fitted scalers.

        Args:
            df: Input DataFrame
            metric_columns: Columns to normalize

        Returns:
            DataFrame with normalized metrics

        """
        if not self.scalers:
            raise ValueError("Scalers must be fitted before transform")

        df_norm = df.clone()
        new_columns = []

        for col in metric_columns:
            if not api.isin_schema(df, column=col) or col not in self.scalers:
                logging.warning(f"Missing column or scaler: {col}")
                continue

            # nearly constant column from fit → keep at 0
            if col in self._constant_columns:
                new_columns.append(pl.Series(col, np.zeros(len(df))))
                continue

            values = df[col].to_numpy().astype(float)

            if self.normalization_type == "custom":
                values = self._custom_transform(values)

            normalized = self.scalers[col].transform(values.reshape(-1, 1))
            new_columns.append(pl.Series(col, normalized.flatten()))

        if new_columns:
            df_norm = df_norm.with_columns(new_columns)

        return df_norm

    def save_scalers(self, path: str | Path) -> None:
        """Save fitted scalers to disk."""
        joblib.dump(
            {
                "scalers": self.scalers,
                "normalization_type": self.normalization_type,
                "scaler_params": self.scaler_params,
                "constant_columns": self._constant_columns,
            },
            path,
        )

    @classmethod
    def load_scalers(cls, path: str | Path) -> "MetricsNormalizer":
        """Create normalizer with pre-fitted scalers.

        Supports StandardScaler, RobustScaler, and custom (arcsinh+RobustScaler).
        The normalization_type is read from the saved file; use the same normalizer
        in inference as was used in preprocessing.

        Transparently loads legacy checkpoints (plain scalers dict)
        by assuming the old default (RobustScaler).
        """
        data = joblib.load(path)

        # legacy format: just the scalers dict, no metadata
        if not isinstance(data, dict) or "normalization_type" not in data:
            normalizer = cls(normalization_type="RobustScaler")
            normalizer.scalers = data
            return normalizer

        normalizer = cls(normalization_type=data["normalization_type"], scaler_params=data.get("scaler_params"))
        normalizer.scalers = data["scalers"]
        normalizer._constant_columns = data.get("constant_columns", set())
        return normalizer


@tl
def prepare_metrics_data(
    df: pl.DataFrame,
    timestamp_col: str,
    metric_columns: list[str],
    anomaly_configs: list[dict[str, Any]],
    time_units: list[str] | None = None,
    normalizer: MetricsNormalizer | None = None,
    normalization_type: str = "StandardScaler",
    scaler_params: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, MetricsNormalizer | None]:
    """Prepare metrics data for training or inference.

    Args:
        df: Input DataFrame
        timestamp_col: Column containing timestamps
        metric_columns: Metrics to normalize
        anomaly_configs: Anomaly filtering configurations
        time_units: Time units for cyclic features
        normalizer: Optional pre-fitted normalizer for inference
        normalization_type: When fitting, use this type (StandardScaler, RobustScaler, custom).
            Must match preprocessing config; inference reads from saved artifact.
        scaler_params: Optional params for the normalizer (e.g. quantile_range for RobustScaler).

    Returns:
        Tuple of (processed DataFrame, normalizer if fitted)

    """
    time_units = time_units or ["hour", "day", "week"]

    # Filter anomalies
    df_filtered = filter_metrics_by_datetime(df, timestamp_col, anomaly_configs)

    # Add cyclic time features
    df_time = compute_cyclic_features(df_filtered, timestamp_col, time_units)

    # Normalize metrics
    if normalizer is None:
        # Training mode - fit new normalizer (use normalization_type from config)
        normalizer = MetricsNormalizer(normalization_type=normalization_type, scaler_params=scaler_params or {})
        df_normalized = normalizer.fit_transform(df_time, metric_columns)
        return df_normalized, normalizer
    else:
        # Inference mode - use existing normalizer
        df_normalized = normalizer.transform(df_time, metric_columns)
        return df_normalized, None
