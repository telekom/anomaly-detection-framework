# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import polars as pl

from datetime import datetime
from pathlib import Path
from typing import Any

from adf.core.common.sys import remove_object
from adf.core.dataframe.base import load_dataframe, save_dataframe

from .steps import MetricsNormalizer, prepare_metrics_data

logger = logging.getLogger(__name__)


class MetricsDataProcessor:
    """Process metrics data for anomaly detection."""

    def __init__(
        self,
        timestamp_col: str = "Time",
        file_format: str = "parquet",
        normalization_type: str = "StandardScaler",
        scaler_params: dict[str, Any] | None = None,
    ):
        """Initialize processor.

        Args:
            timestamp_col: Name of timestamp column
            file_format: Format for data files
            normalization_type: Normalization strategy passed to MetricsNormalizer.
                Options: "StandardScaler", "RobustScaler", "custom"
            scaler_params: Optional dict forwarded directly to MetricsNormalizer.

        """
        self.timestamp_col = timestamp_col
        self.file_format = file_format
        self.normalization_type = normalization_type
        self.scaler_params = scaler_params or {}
        self.normalizer: MetricsNormalizer | None = None

    def split_train_test(
        self,
        input_paths: list[Path],
        output_path: Path,
        cutoff_dates: dict[str, str],
        train_prefix: str = "train",
        val_prefix: str = "val",
        test_prefix: str = "test",
        remove_local: bool = True,
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Split metric data into train, validation and test sets based on cutoff dates.

        Args:
            input_paths: Paths to input files
            output_path: Directory to save split files
            cutoff_dates: Dictionary containing:
                - train_end: Train cutoff date "YYYY-MM-DD"
                - test_start: Test start date "YYYY-MM-DD"
            train_prefix: Prefix for training files
            val_prefix: Prefix for validation files
            test_prefix: Prefix for test files
            remove_local: Whether to remove local files

        Returns:
            tuple[list[Path], list[Path]]: (train_files, test_files)

        """
        train_end = datetime.strptime(cutoff_dates["train_end"], "%Y-%m-%d")
        test_start = datetime.strptime(cutoff_dates["test_start"], "%Y-%m-%d")

        if test_start <= train_end:
            raise ValueError(
                f"Test start date must be after train end date. Got train_end={train_end}, test_start={test_start}"
            )

        train_files, val_files, test_files = [], [], []

        for fp in input_paths:
            try:
                # Load data
                df = load_dataframe(fp, fmt=self.file_format, date_columns=[self.timestamp_col])

                # Split into train/test
                train_df = df.filter(pl.col(self.timestamp_col) <= train_end)
                val_df = df.filter((pl.col(self.timestamp_col) > train_end) & (pl.col(self.timestamp_col) < test_start))
                test_df = df.filter(pl.col(self.timestamp_col) >= test_start)

                # Save train data if not empty
                if len(train_df) > 0:
                    train_path = output_path / f"{train_prefix}_{fp.name}"
                    save_dataframe(train_df, train_path)
                    train_files.append(train_path)

                # Save validation data if not empty
                if len(val_df) > 0:
                    val_path = output_path / f"{val_prefix}_{fp.name}"
                    save_dataframe(val_df, val_path)
                    val_files.append(val_path)

                # Save test data if not empty
                if len(test_df) > 0:
                    test_path = output_path / f"{test_prefix}_{fp.name}"
                    save_dataframe(test_df, test_path)
                    test_files.append(test_path)

                if remove_local:
                    remove_object(fp)

            except Exception as e:
                logging.error(f"Error splitting file {fp}: {e}")
                continue

        logger.info(
            "Split complete - Train files: %d, Validation files: %d, Test files: %d",
            len(train_files),
            len(val_files),
            len(test_files),
        )

        return train_files, val_files, test_files

    def split_and_process(
        self,
        input_paths: list[Path],
        output_path: Path,
        scalers_path: Path,
        metric_columns: list[str],
        cutoff_dates: dict[str, str],
        anomaly_configs: list[dict[str, Any]],
        time_units: list[str] | None = None,
        artifacts_prefix: str = "metrics-artifacts",
        output_prefix: str = "metrics-output",
        remove_local: bool = True,
    ) -> tuple[int, int, int]:
        """Split data into train/val/test and process ensuring proper normalization.

        This method:
        1. Splits data into train/val/test sets
        2. Fits normalizers on train data only
        3. Applies same normalization to val and test data

        Args:
            input_paths: Paths to input files
            output_path: Output directory path
            scalers_path: Path to save fitted scalers
            metric_columns: List of columns to normalize
            cutoff_dates: Dictionary with train_end and test_start dates
            anomaly_configs: Anomaly filtering configurations
            time_units: Time units for cyclic features
            artifacts_prefix: Prefix for model artifacts
            output_prefix: Prefix for processed data
            remove_local: Whether to remove local files

        Returns:
            tuple[int, int, int]: (train_processed_rows, val_processed_rows, test_processed_rows)

        """
        # 1. Split into train/test
        logger.info("Splitting data into train/val/test sets...")
        train_files, val_files, test_files = self.split_train_test(
            input_paths=input_paths,
            output_path=output_path,
            cutoff_dates=cutoff_dates,  # split is based on this cutoff date
            remove_local=False,  # Keep files for processing
        )

        try:
            # 2. Process training data - this will fit the normalizer
            logging.info("Processing training data and fitting normalizers...")
            train_processed = self.process_training_data(
                input_paths=train_files,
                output_path=output_path / "train",
                scalers_path=scalers_path,
                metric_columns=metric_columns,
                anomaly_configs=anomaly_configs,
                time_units=time_units,
                artifacts_prefix=artifacts_prefix,
                output_prefix=f"{output_prefix}/train",
                remove_local=remove_local,
            )

            # 3. Process validation data using fitted normalizer
            logger.info("Processing validation data using fitted normalizers...")
            val_processed = self.process_inference_data(
                input_paths=val_files,
                output_path=output_path / "val",
                scalers_path=scalers_path,  # Use same scalers
                metric_columns=metric_columns,
                time_units=time_units,
                output_prefix=f"{output_prefix}/val",
                remove_local=remove_local,
                anomaly_configs=anomaly_configs,
            )

            # 4. Process test data using fitted normalizer
            logger.info("Processing test data using fitted normalizers...")
            test_processed = self.process_inference_data(
                input_paths=test_files,
                output_path=output_path / "test",
                scalers_path=scalers_path,  # Use same scalers
                metric_columns=metric_columns,
                time_units=time_units,
                output_prefix=f"{output_prefix}/test",
                remove_local=remove_local,
            )

            # Clean up split files if needed
            if remove_local:
                for fp in train_files + test_files:
                    if fp.exists():
                        remove_object(fp)

            return train_processed, val_processed, test_processed

        except Exception as e:
            logging.error(f"Error in split_and_process: {e}")
            # Clean up split files on error
            if remove_local:
                for fp in train_files + val_files + test_files:
                    if fp.exists():
                        remove_object(fp)
            raise

    def process_training_data(
        self,
        input_paths: list[Path],
        output_path: Path,
        scalers_path: Path,
        metric_columns: list[str],
        anomaly_configs: list[dict[str, Any]],
        time_units: list[str] | None = None,
        artifacts_prefix: str = "metrics-artifacts",
        output_prefix: str = "metrics-output",
        remove_local: bool = True,
    ) -> int:
        """Process training data and save artifacts.

        Args:
            input_paths: Paths to input files
            output_path: Output directory path
            scalers_path: Path to save fitted scalers
            metric_columns: Metrics to normalize
            anomaly_configs: Anomaly filtering configurations
            time_units: Time units for cyclic features
            artifacts_prefix: Prefix for model artifacts
            output_prefix: Prefix for processed data
            remove_local: Whether to remove local files

        Returns:
            Number of processed rows

        """
        n_processed = 0

        for fp in input_paths:
            try:
                # Load data
                df = load_dataframe(fp, fmt=self.file_format, date_columns=[self.timestamp_col])

                if remove_local:
                    remove_object(filepath=fp)

                # Process data - fit normalizer on first file (use config normalization_type)
                df_processed, normalizer = prepare_metrics_data(
                    df=df,
                    timestamp_col=self.timestamp_col,
                    metric_columns=metric_columns,
                    anomaly_configs=anomaly_configs,
                    time_units=time_units,
                    normalizer=self.normalizer,
                    normalization_type=self.normalization_type,
                    scaler_params=self.scaler_params,
                )

                # Save processed data
                output_file = output_path / fp.name
                save_dataframe(df_processed, output_file)

                # Save scalers if newly fitted
                if self.normalizer is None and normalizer is not None:
                    self.normalizer = normalizer
                    normalizer.save_scalers(scalers_path)

                n_processed += len(df_processed)

            except Exception as e:
                logging.error(f"Error processing file {fp}: {e}")
                continue

        return n_processed

    def process_inference_data(
        self,
        input_paths: list[Path],
        output_path: Path,
        scalers_path: Path,
        metric_columns: list[str],
        time_units: list[str] | None = None,
        output_prefix: str = "metrics-inference",
        remove_local: bool = True,
        anomaly_configs: list[dict[str, Any]] | None = None,
    ) -> int:
        """Process inference data using saved scalers.

        Args:
            input_paths: Paths to input files
            output_path: Output directory path
            scalers_path: Path to saved scalers
            metric_columns: Metrics to normalize
            time_units: Time units for cyclic features
            output_prefix: Prefix for processed data
            remove_local: Whether to remove local files
            anomaly_configs: Optional anomaly detection configuration overrides

        Returns:
            Number of processed rows

        """
        # Load scalers if not already loaded
        if self.normalizer is None:
            self.normalizer = MetricsNormalizer.load_scalers(scalers_path)

        n_processed = 0

        for fp in input_paths:
            try:
                # Load data
                df = load_dataframe(fp, fmt=self.file_format, date_columns=[self.timestamp_col])

                if remove_local:
                    remove_object(filepath=fp)

                # Process data using saved normalizer
                df_processed, _ = prepare_metrics_data(
                    df=df,
                    timestamp_col=self.timestamp_col,
                    metric_columns=metric_columns,
                    anomaly_configs=anomaly_configs
                    if anomaly_configs is not None
                    else [],  # No filter fr test but fr val
                    time_units=time_units,
                    normalizer=self.normalizer,
                )

                # Save processed data
                output_file = output_path / fp.name
                save_dataframe(df_processed, output_file)

                n_processed += len(df_processed)

            except Exception as e:
                logging.error(f"Error processing file {fp}: {e}")
                continue

        return n_processed

    def save_parameters(
        self,
        total_num: int,
        source_path: str | Path,
        metric_columns: list[str],
        time_units: list[str],
        cutoff_dates: dict[str, str] | None = None,
        anomaly_configs: list[dict[str, Any]] | None = None,
        artifacts_prefix: str = "metrics-artifacts",
        remove_local: bool = True,
    ) -> None:
        """Save preprocessing parameters.

        Args:
            total_num: Total number of processed rows
            source_path: Path to save parameters
            metric_columns: Processed metric columns
            time_units: Time units used for cyclic features
            cutoff_dates: Train/test split dates if used
            anomaly_configs: Anomaly configurations used in training
            artifacts_prefix: Prefix for model artifacts
            remove_local: Whether to remove local files

        """
        storage_uri = ""

        params = {
            "total_num": total_num,
            "storage_uri": storage_uri,
            "timestamp_col": self.timestamp_col,
            "metric_columns": metric_columns,
            "time_units": time_units,
            "cutoff_dates": cutoff_dates,
            "anomaly_configs": anomaly_configs if anomaly_configs else [],
            "file_format": self.file_format,
            "normalization_type": self.normalization_type,
        }

        # Add scaler information if available
        if self.normalizer is not None and self.normalizer.scalers:
            params["normalized_columns"] = list(self.normalizer.scalers.keys())
            params["scaler_type"] = "StandardScaler"
        else:
            params["normalized_columns"] = []
            params["scaler_type"] = None

        # Save parameters
        with open(source_path, "w+") as fp:
            json.dump(params, fp, indent=2, default=str)
