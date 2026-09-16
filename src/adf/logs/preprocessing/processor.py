# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import numpy as np
import polars as pl

from numpy.typing import NDArray
from pathlib import Path
from polars import DataFrame

from adf.core.dataframe.base import load_dataframe, save_dataframe

from .steps import prepare


class DataProcessor:
    """DataProcessor class for processing log data and annotations.

    Attributes:
        incidents (DataFrame | None): Loaded incidents data.

    """

    def __init__(
        self, incidents: str | Path | None = None, *, date_columns: list[str] | None = None, fmt: str = "parquet"
    ):
        """Initialize the DataProcessor with the given file paths and format.

        Args:
            incidents (str): Path to the incidents file.
            date_columns (list[str]): List of date columns in the files. Defaults to None.
            fmt (str): The format of the files (e.g., 'csv', 'json').

        """
        self.incidents = load_dataframe(incidents, date_columns=date_columns, fmt=fmt) if incidents else None
        self.fmt = fmt

    def transform(
        self,
        df: pl.DataFrame | str | Path | list[str] | list[Path],
        *,
        logs_column: str,
        datetime_column: str,
        sequence_length: int,
        sort_more_by: str | None = None,
        event_start_column: str | None = None,
        event_end_column: str | None = None,
        delta_thr: int = 10,
        force_padding_with_values: bool = False,
        padding_constant: int = 0,
        fmt: str = "parquet",
        dirsave: str | Path | None = None,
        outfile: str | None = None,
    ) -> None | tuple[pl.DataFrame, NDArray[np.int_]]:
        """Transform and process input data files.

        Args:
            df (str | Path | list[str] | list[Path]): The input data files to process.
            dirsave (str | None): Directory to save the processed data. If None, data isn't saved.
            outfile (str | None): Name of the output file. If None, data isn't saved.
            logs_column (str): Column name for clean text.
            datetime_column (str): Column name for data time.
            sort_more_by (str | None): Optional additional column name to sort by after sorting by `datetime_column`.
            event_start_column (str): Column name for event start time.
            event_end_column (str): Column name for event end time.
            sequence_length (int): Sequence length for processing.
            delta_thr (int, optional): The minimum number of time units between incidents to be considered consecutive.
                Defaults to 10s.
            force_padding_with_values (bool, optional): Whether to force padding of the window if window_size is smaller
                than its needed from the array. Defaults to False.
            padding_constant (int, optional): Padding constant. Defaults to 0
            fmt (str): The format of the files (e.g., 'csv', 'json', 'parquet').

        Returns:
            np.ndarray: The unique logs in the processed data.

        """
        assert not (dirsave is not None and outfile is None), "If dirsave is provided, outfile must also be provided."

        df = df if isinstance(df, pl.DataFrame) else load_dataframe(df, date_columns=datetime_column, fmt=fmt)

        out = prepare(
            normal=df,
            incidents=self.incidents,
            logs_column=logs_column,
            datetime_column=datetime_column,
            sort_more_by=sort_more_by,
            event_start_column=event_start_column,
            event_end_column=event_end_column,
            sequence_length=sequence_length,
            delta_thr=delta_thr,
            force_padding_with_values=force_padding_with_values,
            padding_constant=padding_constant,
        )

        if dirsave is not None and outfile is not None and out is not None:
            self.save_artifacts(artifacts=out, dirsave=dirsave, outfile=outfile)

        return out

    @staticmethod
    def save_artifacts(artifacts: tuple[DataFrame, NDArray[np.int_]], dirsave: str | Path, outfile: str) -> None:
        """Save artifacts to a directory.

        Args:
            artifacts (list[NDArray[np.int_] | DataFrame]): List of artifacts to save (DataFrames or NumPy arrays).
            dirsave (str | None): Directory path to save the artifacts. If None, artifacts aren't saved.
            outfile (str | None): Name of the output file. If None, artifacts aren't saved.

        Returns:
            None

        """
        if not artifacts:
            logging.warning("No artifacts to save.")
            return

        save_dir = Path(dirsave)
        save_dir.mkdir(parents=True, exist_ok=True)

        for artifact in artifacts:
            if isinstance(artifact, DataFrame):
                save_dataframe(artifact, save_dir / f"{outfile}.parquet", fmt="parquet")
            elif isinstance(artifact, np.ndarray):
                np.save(save_dir / f"{outfile}.npy", artifact)
            else:
                raise TypeError(f"Unsupported type {type(artifact)}")

    @staticmethod
    def save_hparams(source_path: str) -> None:
        """Save hyperparameters to a local file.

        Args:
            source_path (str): Path to save the hyperparameters.

        """
        with open(source_path, "w+") as fp:
            json.dump({}, fp)
