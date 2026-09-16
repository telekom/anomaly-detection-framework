# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import polars as pl

from polars.exceptions import SchemaError
from typing import Literal


def evaluate_log_anomalies(
    df: pl.DataFrame,
    anomalies: pl.DataFrame,
    *,
    event_column: str,
    event_start: str,
    event_end: str,
    threshold: float,
    error: str,
    closed: Literal["both", "left", "right", "none"] = "both",
) -> tuple[int, int, float]:
    """Evaluate anomalies in the dataframe.

    Args:
        df (pl.DataFrame): Dataframe with the data to evaluate.
        anomalies (pl.DataFrame): Dataframe with the anomalies.
        event_column (str): Column with the event in the evaluated dataframe.
        event_start (str): Start of the event in anomalies dataset.
        event_end (str): End of the event in anomalies dataset.
        error (str): Error column in the evaluated dataframe.
        threshold (float): Threshold for the event.
        closed (Literal["both", "left", "right", "none"]): Whether to include the start and end of the event.
            Defaults to "both".

    Returns:
        tp (int): True positives.
        fp (int): False positives.
        prec (float): Precision.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"date": ["2023-01-02", "2023-01-03"], "error": [1, 2]})
        >>> anomalies = pl.DataFrame({"start": ["2023-01-01"], "end": ["2023-01-02"]})
        >>> evaluate_log_anomalies(
        ...     df,
        ...     anomalies,
        ...     event_column="date",
        ...     event_start="start",
        ...     event_end="end",
        ...     error="error",
        ...     threshold=1,
        ...)
        (1, 0, 1.0)

    """
    if error not in df.schema.names() or event_column not in df.schema.names():
        raise SchemaError(f"Columns {error} or {event_column} not found in the dataframe")

    if not isinstance(df.schema[event_column], pl.Datetime):
        raise SchemaError(f"Column '{event_column}' must be of type 'datetime'.")

    if event_start not in anomalies.schema.names() or event_end not in anomalies.schema.names():
        raise SchemaError(f"Columns {event_start} or {event_end} not found in the anomalies dataframe")

    if not isinstance(anomalies.schema[event_start], pl.Datetime) or not isinstance(
        anomalies.schema[event_end], pl.Datetime
    ):
        raise SchemaError(f"Column '{event_start}' and {event_end} must be of type 'datetime'.")

    # Filter the anomalies dataframe to only include the relevant columns
    anomalies = anomalies.select([event_start, event_end])
    df = df.select([event_column, error])

    # Filter dataframe based on the threshold
    df = df.filter(pl.col(error) >= threshold)

    # Exclude values outside the min and max of the event column
    min_start = anomalies[event_start].min()
    max_end = anomalies[event_end].max()
    df = df.filter(pl.col(event_column).is_between(min_start, max_end, closed=closed))

    joined = df.join_where(
        anomalies, pl.col(event_column) >= pl.col(event_start), pl.col(event_column) <= pl.col(event_end)
    )

    tp = len(joined)
    fp = len(df) - tp
    prec = tp / (tp + fp)

    return tp, fp, prec
