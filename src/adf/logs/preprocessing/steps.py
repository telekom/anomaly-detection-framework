# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import numpy as np
import polars as pl

from datetime import timedelta
from numpy.typing import NDArray

from adf.core.dataframe.api import drop_incident_rows, generate_windows, isin_schema

logger = logging.getLogger(__name__)


def prepare(
    normal: pl.DataFrame,
    incidents: pl.DataFrame | None = None,
    *,
    datetime_column: str,
    sequence_length: int,
    logs_column: str,
    sort_more_by: str | None = None,
    index_column: str = "index",
    delta_thr: int = 10,
    event_start_column: str | None = None,
    event_end_column: str | None = None,
    incident_buffer: timedelta | None = None,
    force_padding_with_values: bool = False,
    padding_constant: int = 0,
) -> None | tuple[pl.DataFrame, NDArray[np.int_]]:
    """Prepare the data for processing.

    This function performs the following steps:
        1. Deduplicates the normal dataframe to contain only unique logs in each timestamp.
        2. Based on the incidents dataframe, it filters the normal dataframe to include only the logs that DO NOT
            occur between the start and end of the incidents.
        3. It generates sliding windows of logs based on the specified sequence length and stores the results in the
            output path.

    Args:
        normal (pl.DataFrame): The main dataframe containing the log's data.
        incidents (pl.DataFrame | None, optional): Dataframe containing incident data. Defaults to None.
        logs_column (str): The name of the column containing log data.
        datetime_column (str): The name of the column containing datetime data.
        sort_more_by (str | None): Optional additional column name to sort by after sorting by `datetime_column`.
        index_column (str): The name of the index column in the dataframe.
        delta_thr (int, optional): The minimum number of time units between incidents to be considered consecutive.
            Defaults to 10s.
        sequence_length (int): The length of the sequence windows to extract.
        event_start_column (str): The name of the column containing event start times.
        event_end_column (str): The name of the column containing event end times.
        incident_buffer (int): The buffer size to add to the incident start and end times to make the incident range
            wider. Defaults to 0.
        force_padding_with_values (bool, optional): Whether to force padding of the window if window_size is smaller
            than its needed from the array. Defaults to False.
        padding_constant (int, optional): Padding constant. Defaults to 0

    Returns:
        None | tuple[pl.DataFrame, NDArray[np.int_]]: Returns a tuple containing the dataframe containing the filtered
        incidents dataframe, and the sliding window dataframe.

    """
    if logs_column is not None and not isin_schema(normal, column=logs_column):
        raise ValueError(f"Column '{logs_column}' not found in the dataframe schema: {normal.schema.names()}")

    if isin_schema(normal, column=index_column):
        raise ValueError(f"Column '{index_column}' already exists in the dataframe schema: {normal.schema.names()}")

    # Drop incidents and add index
    df = drop_incident_rows(
        normal,
        incidents,
        datetime_column=datetime_column,
        logs_column=logs_column,
        sort_more_by=sort_more_by,
        event_start_column=event_start_column,
        event_end_column=event_end_column,
        incident_buffer=incident_buffer,
    ).with_row_index(name=index_column)

    if df.is_empty():
        logger.warning("The dataframe is empty after dropping incident rows. Returning None.")
        return None

    indices, windows_array = generate_windows(
        df=df,
        column=datetime_column,
        sequence_length=sequence_length,
        delta_thr=delta_thr,
        return_indices=True,
        force_padding_with_values=force_padding_with_values,
        padding_constant=padding_constant,
    )

    if windows_array.size == 0:
        logger.warning("The windows array is empty. Returning None.")
        return None

    return df[indices], windows_array
