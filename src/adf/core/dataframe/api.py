# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import polars as pl

from datetime import date, timedelta
from functools import partial
from itertools import chain
from numpy.typing import NDArray
from polars.datatypes.classes import NumericType
from polars.exceptions import SchemaError
from typing import TYPE_CHECKING, Any, Literal, cast

from .base import infer_datetime_dtype_for_casting

if TYPE_CHECKING:
    from sklearn.compose import ColumnTransformer

############
## Checks ##
############


def isin_schema(df: pl.DataFrame, *, column: str) -> bool:
    """Check if the specified column is present in the DataFrame schema.

    Args:
        df (pl.DataFrame): DataFrame to check the column in.
        column (str): Name of the column to check.

    Returns:
        bool: True if the column is present in the schema, False otherwise.

    Examples:
        >>> df = pl.DataFrame({"value": [1, 2, 3]})
        >>> isin_schema(df, "value")
        True
        >>> isin_schema(df, "non_existent")
        False

    """
    return column in df.schema.names()


#####################
## Add new columns ##
#####################


def add_aggregates_to_list(
    df: pl.DataFrame,
    groupby_cols: list[str] | str,
    aggregate_cols: list[str] | str,
    aliases: list[str] | str | None = None,
) -> pl.DataFrame:
    """Aggregate the DataFrame by the specified columns and aggregates the specified columns into a list.

    Args:
        df (pl.DataFrame): The DataFrame to be processed.
        groupby_cols (list[str]): list of column names to group by.
        aggregate_cols (list[str]): list of column names to aggregate into a list.
        aliases (list[str], optional): The name of the output columns to store the aggregated lists. Defaults to None.

    Returns:
        pl.DataFrame: A DataFrame with the specified columns aggregated into a list.

    Examples:
        >>> df = pl.DataFrame(
        ...     {"group": ["A", "A", "B", "B", "B", "C"], "value_1": [2, 1, 3, 2, 1, 3], "value_2": [4, 1, 4, 1, 1, 5]}
        ... )
        >>> add_aggregates_to_list(df, "group", ["value", "value2"])
        shape: (3, 3)
        ┌───────┬───────────┬───────────┐
        │ group ┆ value     ┆ value2    │
        │ ---   ┆ ---       ┆ ---       │
        │ str   ┆ list[i64] ┆ list[i64] │
        ╞═══════╪═══════════╪═══════════╡
        │ B     ┆ [3, 2, 1] ┆ [4, 1, 1] │
        │ A     ┆ [2, 1]    ┆ [4, 1]    │
        │ C     ┆ [3]       ┆ [5]       │
        └───────┴───────────┴───────────┘

    """
    aggregate_cols = aggregate_cols if isinstance(aggregate_cols, list) else [aggregate_cols]
    aliases = aggregate_cols if aliases is None else aliases if isinstance(aliases, list) else [aliases]

    return df.group_by(groupby_cols).agg([pl.col(c).alias(a) for c, a in zip(aggregate_cols, aliases)])


def add_zeros(df: pl.DataFrame, *, column: str, dtype: NumericType | None = None) -> pl.DataFrame:
    """Add a column of zeros to the DataFrame.

    Args:
        df (pl.DataFrame): DataFrame to which the column will be added.
        column (str): Name of the column to be added.
        dtype (NumericType): The data type of the column. Default is None.
            If None, the data type will be inferred from the DataFrame.

    Returns:
        pl.DataFrame: DataFrame with the new column added.

    Examples:
        >>> df = pl.DataFrame({"value": [1, 2, 3]})
        >>> add_zeros(df, "zeros")
        shape: (3, 2)
        ┌───────┬──────┐
        │ value ┆ zeros│
        │ ---   ┆ ---  │
        │ i64   ┆ i64  │
        ├───────┼──────┤
        │ 1     ┆ 0    │
        │ 2     ┆ 0    │
        │ 3     ┆ 0    │
        └───────┴──────┘

    """
    return df.with_columns(pl.lit(0, dtype=dtype).alias(column))


def add_ones(df: pl.DataFrame, *, column: str, dtype: NumericType | None = None) -> pl.DataFrame:
    """Add a column of ones to the DataFrame.

    Args:
        df (pl.DataFrame): DataFrame to which the column will be added.
        column (str): Name of the column to be added.
        dtype (NumericType): The data type of the column. Default is None.
            If None, the data type will be inferred from the DataFrame.

    Returns:
        pl.DataFrame: DataFrame with the new column added.

    Examples:
        >>> df = pl.DataFrame({"value": [1, 2, 3]})
        >>> add_ones(df, "ones")
        shape: (3, 2)
        ┌───────┬─────┐
        │ value ┆ ones│
        │ ---   ┆ --- │
        │ i64   ┆ i64 │
        ├───────┼─────┤
        │ 1     ┆ 1   │
        │ 2     ┆ 1   │
        │ 3     ┆ 1   │
        └───────┴─────┘

    """
    return df.with_columns(pl.lit(1, dtype=dtype).alias(column))


def add_alias_from(df: pl.DataFrame, *, column: str, alias: str) -> pl.DataFrame:
    """Add an alias to the specified column in the DataFrame.

    Args:
        df (pl.DataFrame): DataFrame containing the column.
        column (str): The name of the column to be aliased.
        alias (str): The new alias for the column.

    Returns:
        pl.DataFrame: A DataFrame with the specified column aliased.

    Examples:
        >>> df = pl.DataFrame({"value": [1, 2, 3]})
        >>> add_alias_from(df, "value", "new_value")
        shape: (3, 2)
        ┌───────┬───────────┐
        │ value ┆ new_value │
        │ ---   ┆ ---       │
        │ i64   ┆ i64       │
        ╞═══════╪═══════════╡
        │ 1     ┆ 1         │
        │ 2     ┆ 2         │
        │ 3     ┆ 3         │
        └───────┴───────────┘

    """
    return df.with_columns(pl.col(column).alias(alias))


def add_date_from_str(
    df: pl.DataFrame,
    *,
    column: str | list[str],
    time_zone: str | None = None,
    time_unit: Literal["ns", "us", "ms"] = "ns",
) -> pl.DataFrame:
    """Add a date column to a Polars DataFrame by casting specified columns to datetime.

    Args:
        df (pl.DataFrame): The input Polars DataFrame.
        column (list[str] | str): The name(s) of the column(s) to be cast to datetime.
            Can be a single column name or a list of column names.
        time_zone (Optional[str]): The time zone to use for the datetime conversion. Defaults to None.
        time_unit (Optional[str]): The time unit to use for the datetime conversion (e.g., 'ns', 'us', 'ms', 's').
            Defaults to None.

    Returns:
        pl.DataFrame: A new DataFrame with the specified columns cast to datetime.

    Examples:
        >>> df = pl.DataFrame(
        ...     {
        ...         "start": ["2021-01-01 00:00:00", "2021-01-01 00:00:10"],
        ...         "end": ["2021-01-01 00:00:05", "2021-01-01 00:00:15"],
        ...     }
        ... )
        >>> add_date_from_str(df, ["start", "end"], time_unit="ns")
            shape: (2, 2)
            ┌─────────────────────┬─────────────────────┐
            │ start               ┆ end                 │
            │ ---                 ┆ ---                 │
            │ datetime[ns]        ┆ datetime[ns]        │
            ╞═════════════════════╪═════════════════════╡
            │ 2021-01-01 00:00:00 ┆ 2021-01-01 00:00:05 │
            │ 2021-01-01 00:00:10 ┆ 2021-01-01 00:00:15 │
            └─────────────────────┴─────────────────────┘

    """
    column = column if isinstance(column, list) else [column]

    return df.with_columns(
        [infer_datetime_dtype_for_casting(df.schema, c, time_zone=time_zone, time_unit=time_unit) for c in column]
    )


def add_centered_rolling_sum(
    df: pl.DataFrame,
    *,
    column: str,
    safe_range: int,
    index_column: str = "date_time",
    out_column: str = "count",
    timedelta_unit: str = "s",
    closed: Literal["right", "left", "both", "none"] = "right",
) -> pl.DataFrame:
    """Add a time-centered rolling sum column to the given DataFrame.

    This function computes a rolling sum over a specified time window centered
    around each timestamp in the 'date_time' column. The window size is determined
    by the 'safe_range' parameter.

    The 'safe_range' parameter specifies the range in seconds to define the rolling
    window. The total window size will be 2 * safe_range seconds.

    The interval is defined as (timestamp - safe_range, timestamp + safe_range].

    Args:
        df (pl.DataFrame): The input DataFrame containing the data.
        column (str): The name of the column to perform the rolling sum on.
        safe_range (int): The range in seconds to define the rolling window. The total
                        window size will be 2 * safe_range seconds.
        index_column (str): The name of the column containing the timestamps.
                            Default is 'date_time'.
        out_column (str): The name of the new column containing the rolling sum.
                            Default is 'count'.
        timedelta_unit (str): The unit of time to use for the rolling window.
        closed (str): The side of the interval to make closed. Options are ['right', 'left', 'both', 'none'].

    For more information on the available timedelta units, see the Polars documentation:
    https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.rolling.html

    Returns:
        pl.DataFrame: A new DataFrame with the added rolling sum column named 'count'.

    Examples:
        >>> df = pl.DataFrame(
        ...     {
        ...         "date_time": [
        ...             "2022-01-01 00:00:00",
        ...             "2022-01-01 00:00:10",
        ...             "2022-01-01 00:00:20",
        ...             "2022-01-01 00:00:30",
        ...             "2022-01-01 00:00:40",
        ...         ],
        ...         "value": [1, 2, 3, 4, 5],
        ...     }
        ... )
        >>> add_centered_rolling_sum(df, "value", 10)
        shape: (5, 3)
        ┌─────────────────────┬───────┬───────┐
        │ date_time           ┆ value ┆ count │
        │ ---                 ┆ ---   ┆ ---   │
        │ datetime[ns]        ┆ i64   ┆ i64   │
        ╞═════════════════════╪═══════╪═══════╡
        │ 2022-01-01 00:00:00 ┆ 1     ┆ 3     │
        │ 2022-01-01 00:00:10 ┆ 2     ┆ 5     │
        │ 2022-01-01 00:00:20 ┆ 3     ┆ 7     │
        │ 2022-01-01 00:00:30 ┆ 4     ┆ 9     │
        │ 2022-01-01 00:00:40 ┆ 5     ┆ 5     │
        └─────────────────────┴───────┴───────┘

    """
    return df.with_columns(
        pl.sum(column)
        .rolling(
            index_column=index_column,
            period=f"{2 * safe_range}{timedelta_unit}",
            offset=f"-{safe_range}{timedelta_unit}",
            closed=closed,
        )
        .alias(out_column)
    )


def add_forward_sliding_window(
    df: pl.DataFrame,
    column: str | list[str],
    window_size: int,
    *,
    index_column: str = "index",
    output: str | list[str] | None = None,
    filter_valid_rows: bool = False,
) -> pl.DataFrame:
    """Add a sliding window column to the given DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column (str, list[str]): The name of the column(s) to apply the sliding window to.
        window_size (int): The size of the sliding window.
        index_column (str, optional): The name of the index column. Defaults to "index".
        output (str, list[sttr], optional): The name of the output column(s). Defaults to None.
        filter_valid_rows (bool, optional): Whether to filter out rows that can't create a complete window.
            Defaults to False.

    Returns:
        pl.DataFrame: A new DataFrame with the sliding window column added.

    Examples:
        >>> data = {"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        >>> df = pl.DataFrame(data)
        >>> add_forward_sliding_window(df, "values", 3)
        shape: (10, 2)
        ┌────────┬───────────────────────┐
        │ values ┆ values_sliding_window │
        │ ---    ┆ ---                   │
        │ i64    ┆ list[i64]             │
        ╞════════╪═══════════════════════╡
        │ 1      ┆ [1, 2, 3]             │
        │ 2      ┆ [2, 3, 4]             │
        │ 3      ┆ [3, 4, 5]             │
        │ 4      ┆ [4, 5, 6]             │
        │ 5      ┆ [5, 6, 7]             │
        │ 6      ┆ [6, 7, 8]             │
        │ 7      ┆ [7, 8, 9]             │
        │ 8      ┆ [8, 9, 10]            │
        │ 9      ┆ [9, 10]               │
        │ 10     ┆ [10]                  │
        └────────┴───────────────────────┘
        >>> add_forward_sliding_window(df, "values", 3, filter_valid_rows=True)
        shape: (8, 2)
        ┌────────┬───────────────────────┐
        │ values ┆ values_sliding_window │
        │ ---    ┆ ---                   │
        │ i64    ┆ list[i64]             │
        ╞════════╪═══════════════════════╡
        │ 1      ┆ [1, 2, 3]             │
        │ 2      ┆ [2, 3, 4]             │
        │ 3      ┆ [3, 4, 5]             │
        │ 4      ┆ [4, 5, 6]             │
        │ 5      ┆ [5, 6, 7]             │
        │ 6      ┆ [6, 7, 8]             │
        │ 7      ┆ [7, 8, 9]             │
        │ 8      ┆ [8, 9, 10]            │
        └────────┴───────────────────────┘

    """
    if index_column in df.schema.names():
        raise SchemaError(f"Column '{index_column}' already exists in the DataFrame.")

    df = df.with_row_index(name=index_column)

    if not isinstance(column, (str, list)) or (output is not None and not isinstance(output, (str, list))):
        raise TypeError("The 'column' and 'output' arguments must be either a string, a list of strings, or None.")

    column = [column] if isinstance(column, str) else column
    output = (
        [f"{c}_sliding_window" for c in column] if output is None else ([output] if isinstance(output, str) else output)
    )

    if len(column) != len(output):
        raise ValueError("The length of 'column' and 'output' must be the same.")

    exprs = [
        pl.col(column).rolling(index_column, period=f"{window_size}i", closed="right", offset="-1i").alias(output)
        for column, output in zip(column, output)
    ]

    result = df.with_columns(*exprs)

    if filter_valid_rows:
        list_cols = [c for c, t in result.schema.items() if isinstance(t, pl.datatypes.List)]
        result = result.filter(
            pl.fold(
                acc=pl.lit(True),
                function=lambda acc, x: acc & (x.list.len() == window_size),
                exprs=[pl.col(c) for c in list_cols],
            )
        )

    return result.drop(index_column)


def add_groupby_first(
    df: pl.DataFrame, *, sort_cols: str | list[str], groupby_cols: list[str], descending: bool = True
) -> pl.DataFrame:
    """Sort and group by the DataFrame by the specified, returning the first row of each group.

    Args:
        df (pl.DataFrame): The DataFrame to be processed.
        sort_cols (list[str]): list of column names to sort by.
        groupby_cols (list[str]): list of column names to group by.
        descending (bool, optional): Whether to sort in descending order. Defaults to True.

    Returns:
        pl.DataFrame: A DataFrame with the first row of each group after sorting.

    Examples:
        >>> df = pl.DataFrame({"group": ["A", "A", "B", "B", "B", "C"], "value": [2, 1, 3, 2, 1, 3]})
        >>> add_groupby_first(df, "group", "value")
        shape: (3, 2)
        ┌───────┬───────┐
        │ group ┆ value │
        │ ---   ┆ ---   │
        │ str   ┆ i64   │
        ╞═══════╪═══════╡
        │ A     ┆ 1     │
        │ B     ┆ 1     │
        │ C     ┆ 3     │
        └───────┴───────┘

    """
    return df.sort(sort_cols, descending=descending).group_by(groupby_cols).first()


def add_time_delta_difference(
    df: pl.DataFrame, *, column: str, out_column: str, sort: bool = False, time_unit: str = "s"
) -> pl.DataFrame:
    """Add a column with time differences in consecutive rows of a specified column.

    This functions adds a new column to the DataFrame that contains the difference in time
    between consecutive rows of a specified column, converted to a specified time unit.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column (str): The name of the column to calculate the time difference on.
        out_column (str): The name of the output column to store the time differences.
        sort (bool, optional): Whether to sort the DataFrame by the time column.
        time_unit (str, optional): The unit of time to convert the differences to.
                                Options are 'h' (hours), 'm' (minutes), 's' (seconds),
                                'ms' (milliseconds), 'us' (microseconds), 'ns' (nanoseconds).
                                Default is 's'.

    Returns:
        pl.DataFrame: A new DataFrame with the added column containing the time differences.

    Examples:
        >>> df = pl.DataFrame(
        ...     {
        ...         "date_time": [
        ...             "2022-01-01 00:00:00",
        ...             "2022-01-01 00:00:10",
        ...             "2022-01-01 00:00:20",
        ...             "2022-01-01 00:00:30",
        ...         ]
        ...     }
        ... )
        >>> add_time_delta_difference(df, "date_time", "time_diff", "s")
        shape: (4, 2)
        ┌─────────────────────┬────────────┐
        │ date_time           ┆ time_diff  │
        │ ---                 ┆ ---        │
        │ datetime[ns]        ┆ f64        │
        ├─────────────────────┼────────────┤
        │ 2022-01-01 00:00:00 ┆ null       │
        │ 2022-01-01 00:00:10 ┆ 10.0       │
        │ 2022-01-01 00:00:20 ┆ 10.0       │
        │ 2022-01-01 00:00:30 ┆ 10.0       │
        └─────────────────────┴────────────┘

    Raises:
        ValueError: If an invalid time unit is provided.

    """
    if time_unit == "h":
        div = 3600.0
    elif time_unit == "m":
        div = 60.0
    elif time_unit == "s":
        div = 1.0
    elif time_unit == "ms":
        div = 1e-3
    elif time_unit == "us":
        div = 1e-6
    elif time_unit == "ns":
        div = 1e-9
    else:
        raise ValueError("Invalid time unit. Choose from 'h', 'm', 's', 'ms', 'us', 'ns'.")

    if sort:
        df = df.sort(column)

    return df.with_columns((pl.col(column).diff().dt.total_seconds() / div).alias(out_column))


def add_groups_from_time_delta_differences(
    df: pl.DataFrame, *, column: str, out_column: str, threshold: int, sort: bool = False, time_unit: str = "s"
) -> pl.DataFrame:
    """Add a new column to the DataFrame that groups rows based on time difference between consecutive rows of a column.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column (str): The name of the column to calculate the time difference on.
        out_column (str): The name of the output column to store the group numbers.
        threshold (int): The threshold value to determine the groups.
        sort (bool, optional): Whether to sort the DataFrame by the time column.
        time_unit (str, optional): The unit of time to convert the differences to.
                                Options are 'h' (hours), 'm' (minutes), 's' (seconds),
                                'ms' (milliseconds), 'us' (microseconds), 'ns' (nanoseconds).
                                Default is 's'.

    Returns:
        pl.DataFrame: A new DataFrame with the added column containing the group numbers.

    Examples:
        >>> df = pl.DataFrame(
        ...     {
        ...         "date_time": [
        ...             "2022-01-01 00:00:00",
        ...             "2022-01-01 00:00:04",
        ...             "2022-01-01 00:00:20",
        ...             "2022-01-01 00:00:30",
        ...             "2022-01-01 00:00:40",
        ...         ],
        ...         "value": [1, 2, 3, 4, 5],
        ...     }
        ... )
        >>> add_groups_from_time_delta_differences(df, "date_time", "time_group", 5, "s")
        shape: (5, 3)
        ┌─────────────────────┬───────┬─────────────┐
        │ date_time           ┆ value ┆ time_groups │
        │ ---                 ┆ ---   ┆ ---         │
        │ datetime[ns]        ┆ i64   ┆ u32         │
        ╞═════════════════════╪═══════╪═════════════╡
        │ 2022-01-01 00:00:00 ┆ 1     ┆ 0           │
        │ 2022-01-01 00:00:04 ┆ 2     ┆ 0           │
        │ 2022-01-01 00:00:20 ┆ 3     ┆ 1           │
        │ 2022-01-01 00:00:30 ┆ 4     ┆ 2           │
        │ 2022-01-01 00:00:40 ┆ 5     ┆ 3           │
        └─────────────────────┴───────┴─────────────┘

    """
    df = add_time_delta_difference(df, column=column, out_column=out_column, sort=sort, time_unit=time_unit)

    df = df.with_columns(
        (pl.col(out_column) > threshold)
        .cum_sum()
        .fill_null(strategy="zero")  # Fill the first row with 0 to make sure it is included in the first group
        .alias(out_column)
    )

    return df


def add_replaced_string(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    column: str,
    pattern: str | pl.Expr,
    replacement: str | pl.Expr,
    out_column: str | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Add a new column to the DataFrame that maps strings based on a specified pattern and replacement.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column (str): The name of the column to map strings on.
        pattern (list[str]): The list of patterns to match in the strings.
        replacement (list[str]): The list of replacements for the matched patterns.
        out_column (str, optional): The name of the output column to store the mapped strings. Defaults to None.

    Returns:
        pl.DataFrame: A new DataFrame with the added column containing the mapped strings.

    Examples:
        >>> df = pl.DataFrame({"text": ["ab12cd34ef", "gh45ij67kl"]})
        >>> df = add_replaced_string(df, "text", r"(?<N>{2,})", "$N", "text1")
        >>> df = add_replaced_string(df, "text", r"(?<N>{2,})", "$$N$$", "text2")
        shape: (2, 3)
        ┌────────────┬──────────────┬──────────────┐
        │ text       ┆ text1        ┆ text2        │
        │ ---        ┆ ---          ┆ ---          │
        │ str        ┆ str          ┆ str          │
        ╞════════════╪══════════════╪══════════════╡
        │ ab12cd34ef ┆ ab12$cd34$ef ┆ ab$N$cd$N$ef │
        │ gh45ij67kl ┆ gh45$ij67$kl ┆ gh$N$ij$N$kl │
        └────────────┴──────────────┴──────────────┘

    """
    out_column = out_column or f"{column}_mapped"
    return df.with_columns(pl.col(column).str.replace_all(pattern, replacement).alias(out_column))


def add_cyclic_features(df: pl.DataFrame, time_column: str, periods: dict[str, float] | None = None) -> pl.DataFrame:
    """Create cyclic time features (hour, day, week) from timestamps using sine and cosine transformations.

    Args:
        df (pl.DataFrame): The input DataFrame.
        time_column (str): The column containing the timestamps.
        periods (dict[str, int], optional): The periods for the cyclic features. Defaults to None.

    Returns:
        pl.DataFrame: A new DataFrame with the added cyclic features.

    Examples:
        >>> df = pl.DataFrame(
        ...     {
        ...         "timestamp": [
        ...             "2022-01-01 00:00:00",
        ...             "2022-01-01 12:00:00",
        ...             "2022-01-02 00:00:00",
        ...             "2022-01-02 12:00:00",
        ...         ]
        ...     }
        ... ).with_columns(pl.col("timestamp").str.to_datetime())
        >>> add_cyclic_features(df, "timestamp")
        shape: (4, 7)
        ┌──────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
        │ timestamp    ┆ timestamp_h ┆ timestamp_h ┆ timestamp_d ┆ timestamp_d ┆ timestamp_w ┆ timestamp_w │
        │ ---          ┆ our_cos     ┆ our_sin     ┆ ay_cos      ┆ ay_sin      ┆ eek_cos     ┆ eek_sin     │
        │ datetime[μs] ┆ ---         ┆ ---         ┆ ---         ┆ ---         ┆ ---         ┆ ---         │
        │              ┆ f64         ┆ f64         ┆ f64         ┆ f64         ┆ f64         ┆ f64         │
        ╞══════════════╪═════════════╪═════════════╪═════════════╪═════════════╪═════════════╪═════════════╡
        │ 2022-01-01   ┆ 1.0         ┆ -4.4049e-10 ┆ 1.0         ┆ -8.6523e-12 ┆ -0.222521   ┆ 0.974928    │
        │ 00:00:00     ┆             ┆             ┆             ┆             ┆             ┆             │
        │ 2022-01-01   ┆ 1.0         ┆ -5.6353e-10 ┆ -1.0        ┆ 1.3779e-11  ┆ -0.62349    ┆ 0.781831    │
        │ 12:00:00     ┆             ┆             ┆             ┆             ┆             ┆             │
        │ 2022-01-02   ┆ 1.0         ┆ -6.8657e-10 ┆ 1.0         ┆ -1.8906e-11 ┆ -0.900969   ┆ 0.433884    │
        │ 00:00:00     ┆             ┆             ┆             ┆             ┆             ┆             │
        │ 2022-01-02   ┆ 1.0         ┆ 1.2172e-10  ┆ -1.0        ┆ -5.0715e-12 ┆ -1.0        ┆ -7.2450e-13 │
        │ 12:00:00     ┆             ┆             ┆             ┆             ┆             ┆             │
        └──────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘

    """
    second_to_units = {"ns": 1e9, "us": 1e6, "ms": 1e3}  # How many time units a second has

    if not isinstance(time_column, str):
        raise TypeError("Invalid type. Must be a string.")

    if time_column not in df.schema.names():
        raise SchemaError(f"Column '{time_column}' not found in the DataFrame.")

    if not isinstance(df.schema[time_column], pl.Datetime):
        raise SchemaError(f"Column '{time_column}' must be of type 'datetime'.")

    time_unit_converter = second_to_units[df.schema[time_column].time_unit]
    periods = periods or {
        "hour": 60 * 60 * time_unit_converter,
        "day": 24 * 60 * 60 * time_unit_converter,
        "week": 7 * 24 * 60 * 60 * time_unit_converter,
    }

    def _add_cyclic(timestamp: str | pl.Series | NDArray[Any], period: float, name: str) -> list[pl.Expr]:
        return [
            (pl.col(timestamp).cast(pl.Float64) * 2 * np.pi / period).cos().alias(f"{timestamp}_{name}_cos"),
            (pl.col(timestamp).cast(pl.Float64) * 2 * np.pi / period).sin().alias(f"{timestamp}_{name}_sin"),
        ]

    exprs = list(chain(*[_add_cyclic(time_column, period, name) for name, period in periods.items()]))
    return df.with_columns(exprs)


######################
## Modify dataframe ##
######################


def scale(
    df: pl.DataFrame,
    column: str | list[str],
    *,
    fitted_transformer: ColumnTransformer | None = None,
    return_transformer: bool = False,
    base_transform: str = "minmax",
    per_column_transforms: dict[str, str] | None = None,
    remainder: Literal["passthrough", "drop"] = "passthrough",
    verbose_feature_names_out: bool = False,
    **kwargs: Any,
) -> pl.DataFrame | tuple[pl.DataFrame, ColumnTransformer]:
    """Add normalized columns to the dataframe using pre-fitted scalars on training data.

    Args:
        df (pl.DataFrame): Input dataframe.
        column (str | list[str]): Column(s) to normalize.
        fitted_transformer (ColumnTransformer, optional): Pre-fitted transformer for transforming only.
            Defaults to None.
        return_transformer (bool, optional): Return the transformer object for further use. Defaults to False.
        base_transform (str, optional): Base transform to use. Defaults to "minmax".
        per_column_transforms (dict[str, str], optional): Per column transforms. Defaults to None.
        remainder (Literal["passthorugh", "drop"], optional): What to do with the remaining columns.
            Defaults to "passthrough".
        verbose_feature_names_out (bool, optional): Whether to add prefix to feature names. Defaults to False.
        **kwargs: Additional keyword arguments for the transformer.

    Returns:
        tuple[pl.DataFrame | ColumnTransformer] | pl.DataFrame: Normalized dataframe and or ColumnTransformer.

    Examples:
        >>> import polars as pl
        >>> from sklearn.preprocessing import MinMaxScaler
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "c": [1, 2, 3, 4, 5]})
        >>> df = scale(df, column=["a", "b"], return_transformer=False)
        >>> df
        shape: (5, 3)
        ┌──────┬──────┬─────┐
        │ a    ┆ b    ┆ c   │
        │ ---  ┆ ---  ┆ --- │
        │ f64  ┆ f64  ┆ i64 │
        ╞══════╪══════╪═════╡
        │ 0.0  ┆ 1.0  ┆ 1   │
        │ 0.25 ┆ 0.75 ┆ 2   │
        │ 0.5  ┆ 0.5  ┆ 3   │
        │ 0.75 ┆ 0.25 ┆ 4   │
        │ 1.0  ┆ 0.0  ┆ 5   │
        └──────┴──────┴─────┘

    """
    from sklearn.compose import ColumnTransformer as _ColumnTransformer
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    def _get_base_transform(transform: str, **kwargs: Any) -> Any:
        """Get the base transform object based on the name."""
        if transform == "minmax":
            return MinMaxScaler(**kwargs)
        elif transform == "standard":
            return StandardScaler(**kwargs)
        elif transform == "robust":
            return RobustScaler(**kwargs)
        else:
            raise ValueError(f"Unknown transform: {transform}. Available: minmax, standard, robust")

    if fitted_transformer is not None:
        if not isinstance(fitted_transformer, _ColumnTransformer):
            raise TypeError("Invalid transformer object.")

        if (
            not hasattr(fitted_transformer, "_sklearn_output_config")
            and fitted_transformer._sklearn_output_config["transform"] != "polars"
        ):
            raise AttributeError("Transformer sklearn output config not set to polars.")

        return cast(pl.DataFrame, fitted_transformer.transform(df))

    if isinstance(column, str):
        column = [column]

    per_column_transforms = per_column_transforms or {}

    cts = {}

    for c in column:
        if c not in df.columns:
            raise ValueError(f"Column {c} not found in dataframe.")

        if c in per_column_transforms:
            ct = per_column_transforms[c]
            cts.setdefault(ct, []).append(c)
        else:
            ct = base_transform
            cts.setdefault(ct, []).append(c)

    transformer = _ColumnTransformer(
        transformers=[(ct, _get_base_transform(ct, **kwargs), cts[ct]) for ct in cts],
        remainder=remainder,
        verbose_feature_names_out=verbose_feature_names_out,
        force_int_remainder_cols=False,
    )

    transformer.set_output(transform="polars")
    df = transformer.fit_transform(df)

    if return_transformer:
        return df, transformer
    else:
        return df


def drop_shortest_interval(df: pl.DataFrame, end: str, start: str) -> pl.DataFrame:
    """Remove subintervals with the same start time and keeps only the longest one.

    Args:
        df (pl.DataFrame): DataFrame containing the log data.
        start (str): The column name representing the start datetime.
        end (str): The column name representing the end datetime.

    Returns:
        pl.DataFrame: A DataFrame with subintervals removed, keeping only the longest.

    Examples:
        >>> df = pl.DataFrame(
        ...     {
        ...         "start": ["2021-01-01 00:00:00", "2021-01-01 00:00:00"],
        ...         "end": ["2021-01-01 00:00:05", "2021-01-01 00:00:10"],
        ...         "value": [1, 2],
        ...     }
        ... )
        >>> drop_shortest_interval(df, "end", "start")
        shape: (1, 3)
        ┌────────────────────┬────────────────────┬───────┐
        │ start              ┆ end                ┆ value │
        │ ---                ┆ ---                ┆ ---   │
        │ datetime           ┆ datetime           ┆ i64   │
        ├────────────────────┼────────────────────┼───────┤
        │ 2021-01-01 00:00:00┆ 2021-01-01 00:00:10┆ 2     │
        └────────────────────┴────────────────────┴───────┘

    """
    return df.sort([start, end]).group_by(start).agg(pl.all().exclude(start).last()).sort(start)


def split_intervals(
    df: pl.DataFrame,
    *,
    start: str,
    end: str,
    safe_range: int,
    interval_unit_t: str = "s",
    closed: Literal["left", "right", "both", "none"] = "left",
) -> pl.DataFrame:
    """Replicate rows in the DataFrame by incrementing the datetime values within a safe range.

    Args:
        df (pl.DataFrame): DataFrame containing the log data.
        start (str): The column name representing the start datetime.
        end (str): The column name representing the end datetime.
        safe_range (int): The range within which to increment the datetime values.
        interval_unit_t (str): The unit of time for the interval. Defaults to "s".
        closed (str): The interval closure type. Defaults to "left".

    Returns:
        pl.DataFrame: A DataFrame with replicated rows based on the specified safe range.

    Examples:
        >>> pl.DataFrame(
        ...     {
        ...         "start": ["2021-01-01 00:00:00", "2021-01-01 00:00:05"],
        ...         "end": ["2021-01-01 00:00:10", "2021-01-01 00:00:15"],
        ...         "data_index": [1, 2],
        ...     }
        ... )
        >>> split_intervals(df, "start", "end", 5)
        shape: (4, 3)
        ┌─────────────────────┬─────────────────────┬────────────┐
        │ start               ┆ end                 ┆ data_index │
        │ ---                 ┆ ---                 ┆ ---        │
        │ datetime[μs]        ┆ datetime[μs]        ┆ i64        │
        ╞═════════════════════╪═════════════════════╪════════════╡
        │ 2021-01-01 00:00:00 ┆ 2021-01-01 00:00:05 ┆ 1          │
        │ 2021-01-01 00:00:05 ┆ 2021-01-01 00:00:10 ┆ 1          │
        │ 2021-01-01 00:00:05 ┆ 2021-01-01 00:00:10 ┆ 2          │
        │ 2021-01-01 00:00:10 ┆ 2021-01-01 00:00:15 ┆ 2          │
        └─────────────────────┴─────────────────────┴────────────┘

    """
    return df.with_columns(
        pl.datetime_ranges(start, end, interval=f"{safe_range}{interval_unit_t}", closed=closed).alias(start)
    ).explode(start)


def split_longest_interval(df: pl.DataFrame, *, start: str, end: str, safe_range: int) -> pl.DataFrame:
    """Divides the longest interval in the DataFrame into multiple intervals based on the safe range.

    Args:
        df (pl.DataFrame): DataFrame containing the log data.
        start (str): The column name representing the start datetime.
        end (str): The column name representing the end datetime.
        safe_range (int): The range within which to increment the datetime values.

    Returns:
        pl.DataFrame: A DataFrame with the longest interval divided into multiple intervals.

    Examples:
        >>> df = pl.DataFrame(
        ...     {
        ...         "start": ["2021-01-01 00:00:00", "2021-01-01 00:00:05"],
        ...         "end": ["2021-01-01 00:00:00", "2021-01-01 00:00:15"],
        ...         "value": [1, 2],
        ...     }
        ... )
        >>> split_longest_interval(df, "start", "end", 5)
        shape: (3, 3)
        ┌─────────────────────┬─────────────────────┬────────────┐
        │ start               ┆ end                 ┆ data_index │
        │ ---                 ┆ ---                 ┆ ---        │
        │ datetime[μs]        ┆ datetime[μs]        ┆ i64        │
        ╞═════════════════════╪═════════════════════╪════════════╡
        │ 2021-01-01 00:00:05 ┆ 2021-01-01 00:00:15 ┆ 2          │
        │ 2021-01-01 00:00:10 ┆ 2021-01-01 00:00:15 ┆ 2          │
        │ 2021-01-01 00:00:15 ┆ 2021-01-01 00:00:15 ┆ 2          │
        └─────────────────────┴─────────────────────┴────────────┘

    """
    return df.pipe(drop_shortest_interval, end=end, start=start).pipe(
        split_intervals, start=start, end=end, safe_range=safe_range
    )


def lists_to_arrays(df: pl.DataFrame, columns: str | list[str], size: int) -> pl.DataFrame:
    """Convert list columns to array columns in the DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        columns (list[str]): List of column names to convert.
        size (int): The size of the arrays.

    Returns:
        pl.DataFrame: DataFrame with the specified columns converted to arrays.

    Examples:
        >>> df = pl.DataFrame({"values": [[1, 2, 3], [4, 5, 6]]})
        >>> lists_to_arrays(df, ["values"], 3)
        shape: (2, 1)
        ┌────────────────┐
        │ values         │
        │ ---            │
        │ array[i64, 3]  │
        ╞════════════════╡
        │ [1, 2, 3]      │
        │ [4, 5, 6]      │
        └────────────────┘

    """
    if not isinstance(columns, str | list):
        raise TypeError("Columns must be a string or a list of strings.")

    if not isinstance(columns, list):
        columns = [columns]

    for c in columns:
        if not isinstance(df.schema[c], pl.datatypes.List):
            raise pl.exceptions.SchemaError(f"Column '{c}' is not a list.")

    exprs = [pl.col(c).list.to_array(size) for c in columns]
    return df.with_columns(*exprs)


######################
## Filter dataframe ##
######################


def filter_datetime_range(
    df: pl.DataFrame, *, start: str, end: str, dtime_min: timedelta | date | Any, dtime_max: timedelta | date | Any
) -> pl.DataFrame:
    """Filter rows in the DataFrame that fall within the specified datetime range.

    It assumes `start` and `end` columns are of type `datetime64[ns]`.

    Args:
        df (pl.DataFrame): DataFrame containing the log data.
        start (str): The column name representing the start datetime.
        end (str): The column name representing the end datetime.
        dtime_min (datetime): The minimum datetime for the filter range.
        dtime_max (datetime): The maximum datetime for the filter range.

    Returns:
        pl.DataFrame: A DataFrame with rows filtered based on the specified datetime range.

    Examples:
        >>> df = pl.DataFrame(
        ...     {
        ...         "start": ["2021-01-01 00:00:00", "2021-01-01 00:00:10"],
        ...         "end": ["2021-01-01 00:00:05", "2021-01-01 00:00:15"],
        ...         "value": [1, 2],
        ...     }
        ... )
        >>> dtime_min = datetime(2021, 1, 1, 0, 0, 0)
        >>> dtime_max = datetime(2021, 1, 1, 0, 0, 10)
        >>> filter_datetime_range(df, "start", "end", dtime_min, dtime_max)
        shape: (1, 3)
        ┌────────────────────┬────────────────────┬───────┐
        │ start              ┆ end                ┆ value │
        │ ---                ┆ ---                ┆ ---   │
        │ datetime           ┆ datetime           ┆ i64   │
        ├────────────────────┼────────────────────┼───────┤
        │ 2021-01-01 00:00:00┆ 2021-01-01 00:00:05┆ 1     │
        └────────────────────┴────────────────────┴───────┘

    """
    return df.filter((pl.col(end) > dtime_min) & (pl.col(start) < dtime_max))


def filter_group_length(
    df: pl.DataFrame, *, groupby_col: str, min_length: int, alias: str | None = None
) -> pl.DataFrame:
    """Filter groups in the DataFrame based on the length of each group.

    Args:
        df (pl.DataFrame): DataFrame containing the data.
        groupby_col (str): The column name to group by.
        min_length (int): The minimum length of the groups to keep.
        alias (str, optional): The alias to use for the length column. Defaults to None.

    Returns:
        pl.DataFrame: A DataFrame with groups filtered based on the specified length.

    Examples:
        >>> df = pl.DataFrame({"group": ["A", "A", "B", "B", "B", "C"], "value": [1, 2, 3, 2, 1, 3]})
        >>> filter_group_length(df, "group", 2)
        shape: (2, 1)
        ┌───────┐
        │ group │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ A     │
        │ B     │
        └───────┘

    """
    alias = "len" if alias is None else alias

    return (df.group_by(groupby_col).len(name=alias).filter(pl.col(alias) >= min_length)).drop(alias)


############
## Events ##
############


def drop_incident_rows(
    df: pl.DataFrame,
    incidents: pl.DataFrame | None = None,
    *,
    datetime_column: str,
    logs_column: str,
    sort_more_by: str | None = None,
    event_start_column: str | None = None,
    event_end_column: str | None = None,
    incident_buffer: timedelta | None = None,
) -> pl.DataFrame:
    """Drop rows from the DataFrame that fall within the incident intervals.

    This function filters out rows from the input DataFrame that fall within the intervals defined by the incident
    DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame containing logs.
        incidents (pl.DataFrame): The DataFrame containing incident intervals.
        event_start_column (str): The column name representing the start datetime in the incident DataFrame.
        event_end_column (str): The column name representing the end datetime in the incident DataFrame.
        datetime_column (str): The column name representing the datetime in the input DataFrame.
        logs_column (str): The column name representing the logs in the input DataFrame.
        sort_more_by (str | None): Optional additional column name to sort by after sorting by `datetime_column`.
        incident_buffer (timedelta | None): Optional buffer time to extend the incident intervals.

    Returns:
        pl.DataFrame: A DataFrame with rows filtered based on the specified incident intervals.

    """
    if incidents is None:
        if sort_more_by is None:
            return df.sort(datetime_column, maintain_order=True)

        return df.sort(datetime_column, sort_more_by, maintain_order=True)

    if not event_end_column or not event_start_column:
        raise ValueError("event_end_column and event_start_column must be specified")

    if not isinstance(incidents, pl.DataFrame):
        raise TypeError("Invalid type. Incidents must be a DataFrame.")

    if incidents is not None and not all([event_start_column, event_end_column]):
        raise ValueError("If incidents are provided, both event_start_column and event_end_column must be specified.")

    incidents = incidents.with_columns(
        [
            pl.col(event_start_column) - incident_buffer if incident_buffer else pl.col(event_start_column),
            pl.col(event_end_column) + incident_buffer if incident_buffer else pl.col(event_end_column),
        ]
    )

    # Drop rows that fall within the incident intervals
    df = (
        df.join_where(
            incidents,
            ~(pl.col(datetime_column) > pl.col(event_start_column))
            & (pl.col(datetime_column) < pl.col(event_end_column)),
        )
        .unique([datetime_column, logs_column], maintain_order=True)
        .drop([event_start_column, event_end_column])
    )

    if sort_more_by is None:
        return df.sort(datetime_column, maintain_order=True)

    return df.sort(datetime_column, sort_more_by, maintain_order=True)


###############
## Windowing ##
###############


def generate_windows(
    df: pl.DataFrame,
    *,
    column: str,
    delta_thr: int = 10,
    start_index: int = 0,
    index_name: str = "index",
    sequence_length: int = 2,
    return_indices: bool = True,
    force_padding_with_values: bool = False,
    padding_constant: int = 0,
) -> tuple[list[int], NDArray[np.int_]] | NDArray[np.int_]:
    """Return indices for split slices of a DataFrame based on time delta thresholds and sequence length constraints.

    This function processes a sorted Polars DataFrame, grouping rows on the time difference between consecutive rows.
    Rows belong to the same group if the time difference is less than or equal to delta_thr seconds.
    If then filters the groups on a minimum sequence length.
    Then it and creates sliding windows of sequence_length on every group.
    Finally, it returns the indices of the valid groups and the corresponding sliding windows as a 2D numpy array.

    Args:
        df (pl.DataFrame):
            A sorted Polars DataFrame containing a timestamp column and an index column.
        delta_thr (int):
            Maximum allowable time difference (in seconds) between consecutive rows to consider them part same group.
        column (str):
            The name of the column containing the timestamps to compute time deltas.
        start_index (int):
            The starting index for a new column to track processed indices.
        sequence_length (int):
            The minimum required sequence length for each slice.
        index_name (str, optional):
            The name of the column containing the original indices. Defaults to a global `IDX_COL_NAME`.
        return_indices (bool, optional):
            Whether to return the indices along with the sliding windows. Defaults to True
        force_padding_with_values (bool, optional):
            Whether to add zeros in the window if the window is smaller than sequence_length. Defaults to False.
        padding_constant (int, optional):
            The value to pad the windows with. Defaults to 0.

    Returns:
        tuple[list[int], NDArray]:
            - A list of all valid indices that satisfy the time delta and sequence length conditions.
            - A 2D NumPy array where each row represents a sliding window of indices.

    Examples:
        >>> df = pl.DataFrame(
        ...     {
        ...         "date_time": [
        ...             "2022-01-01 00:00:00",
        ...             "2022-01-01 00:00:10",
        ...             "2022-01-01 00:00:20",
        ...             "2022-01-01 00:00:30",
        ...             "2022-01-01 00:00:40",
        ...         ],
        ...         "value": [1, 2, 3, 4, 5],
        ...     }
        ... )
        >>> generate_windows(df, delta_thr=10, column="date_time", curr_max_ind=0, sequence_length=2)
        ([0, 1, 2, 3, 4],
        array([[0, 1],
                [1, 2],
                [2, 3],
                [3, 4]], dtype=uint32))
        >>> generate_windows(df, delta_thr=10, column="date_time", curr_max_ind=0, sequence_length=3))
        ([0, 1, 2, 3, 4],
        array([[0, 1, 2],
                [1, 2, 3],
                [2, 3, 4]], dtype=uint32)
        >>> generate_windows(df, delta_thr=5, column="date_time", curr_max_ind=0, sequence_length=2)
        ([], array([], shape=(0, 3), dtype=int64))
        >>> generate_windows(df, delta_thr=10, column="date_time", sequence_length=8, force_padding_with_values=True)
        ([1, 2, 3, 4, 5], array([[1, 2, 3, 4, 5, 0, 0, 0]], dtype=uint32))

    """
    if not isin_schema(df, column=index_name):
        raise SchemaError(f"Column '{index_name}' not found in the DataFrame. Specify the dataset's index column.")

    from adf.core.common.arrays import sliding_window

    fn = partial(
        sliding_window,
        window_size=sequence_length,
        force_padding_with_values=force_padding_with_values,
        padding_constant=padding_constant,
    )

    # Create the time delta differenced groups
    df = add_groups_from_time_delta_differences(
        df=df, column=column, out_column="_delta_group", time_unit="s", threshold=delta_thr
    )

    # If not padding - drop groups that are smaller than sequence_length
    if force_padding_with_values:
        valid_groups = df.select("_delta_group").unique()
    else:
        valid_groups = filter_group_length(df=df, groupby_col="_delta_group", min_length=sequence_length)

    # Join the valid groups back to the original DataFrame and set a new index column to track the original order
    df = df.join(valid_groups, on="_delta_group", how="inner")

    # Add a new index column to the DataFrame
    df = df.with_row_index(name="_delta_idx", offset=start_index)

    # Get the indices and windows
    inds = df.select(index_name).to_numpy().flatten().tolist()

    # Replace df with the grouped DataFrame and aggregate the indices into a list
    df = add_aggregates_to_list(
        df=df, groupby_cols="_delta_group", aggregate_cols="_delta_idx", aliases="_delta_group_indices"
    )

    arr_windows = np.fromiter(
        (fn(x) for x in df.select("_delta_group_indices").to_numpy().squeeze(1)), dtype=np.ndarray
    )

    arr_windows = np.vstack(arr_windows) if arr_windows.size > 0 else np.empty((0, sequence_length), dtype=int)  # type: ignore

    if return_indices:
        return inds, arr_windows

    return arr_windows
