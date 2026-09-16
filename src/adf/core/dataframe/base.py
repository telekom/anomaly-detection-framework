# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import polars as pl

from collections.abc import Callable
from pathlib import Path
from polars._typing import FrameInitTypes
from polars.exceptions import PolarsError
from typing import Any, Literal

#################
## Save / Load ##
#################


def load_dataframe(
    filepath: str | Path | list[str] | list[Path],
    *,
    fmt: str = "parquet",
    date_columns: str | list[str] | None = None,
    reader_config: dict[str, Any] | None = None,
    dates_config: dict[str, Any] | None = None,
    lazy: bool = False,
) -> pl.DataFrame:
    """Load a file into a DataFrame, with optional date column parsing and custom reader arguments.

    Args:
        filepath (str | Path): The path to the file to be loaded.
        date_columns (list[str] | None): list of columns to be parsed as dates. Defaults to None.
        fmt (str): The file format to load. Defaults to "parquet".
        reader_config (dict): Additional configuration options for the reader. Defaults to an empty dictionary.
        dates_config (dict): Additional keyword arguments for the date parser. Defaults to an empty dictionary.
        lazy (bool): Indicates if the dataframe will be a Polars LazyFrame. Defaults to False.

    Returns:
        DataFrame: A Polars DataFrame loaded from the specified file.

    Examples:
        >>> load_dataframe("data.parquet", date_columns=["date"])

    """
    from .api import add_date_from_str  # imported here to avoid circular imports

    reader_config = reader_config or dict()
    dates_config = dates_config or dict()

    reader = create_polars_reader(fmt, lazy_frame=lazy)

    try:
        df = reader(filepath, **reader_config)
        if date_columns is not None:
            df = add_date_from_str(df, column=date_columns, **dates_config)
        return df
    except Exception as e:
        raise e


def create_dataframe(
    data: FrameInitTypes | pl.DataFrame, date_columns: str | list[str] | None = None, **kwargs: Any
) -> pl.DataFrame:
    """Create a Polars DataFrame from the given data.

    Args:
        data (dict[str, list] | pl.DataFrame]): The data to be converted to a DataFrame.
        date_columns (str | list[str] | None): The column(s) to be parsed as dates. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the DataFrame constructor.

    Returns:
        pl.DataFrame: A Polars DataFrame created from the given data.

    Examples:
        >>> data = {"value": [1, 2, 3]}
        >>> create_dataframe(data)

    """
    from .api import add_date_from_str  # imported here to avoid circular imports

    try:
        df = pl.DataFrame(data, **kwargs)
    except Exception as e:
        raise PolarsError(f"Not supported data type: {type(data)}") from e

    if date_columns is not None:
        return add_date_from_str(df, column=date_columns, **kwargs)

    return df


def save_dataframe(
    data: FrameInitTypes | pl.DataFrame, destination: str | Path, fmt: str = "parquet", **kwargs: Any
) -> None:
    """Create a Polars DataFrame from the given data and write it to a file in the specified format.

    Args:
        data (dict[str, list] | pl.DataFrame]): The data to be written to the file.
        destination (str | Path): The path to the file to be written.
        fmt (str): The format of the file to be written. Defaults to "parquet".
        **kwargs: Additional keyword arguments to be passed to the DataFrame constructor.

    Examples:
        >>> data = {"value": [1, 2, 3]}
        >>> save_dataframe(data, "data.parquet")

    """
    writers = {x.split("_")[1]: x for x in dir(pl.DataFrame) if x.startswith("write_")}

    if fmt not in writers and fmt != "npy":
        raise PolarsError(f"Unsupported file format: {fmt}")

    if isinstance(data, pl.DataFrame):
        df = data
    else:
        try:
            df = pl.DataFrame(data, **kwargs)
        except Exception as e:
            raise PolarsError(f"Not supported data type: {type(data)}") from e

    if fmt == "npy":
        np.save(destination, df.to_numpy())
    else:
        getattr(df, writers[fmt])(destination)


def save_dataframe_with_auto_increment(data: pl.DataFrame, path: Path, base_filename: str = "dataset") -> None:
    """Save the given dataset as a Parquet file in the specified directory.

    If the directory does not exist, it will be created, and the dataset
    will be saved with the specified base filename followed by "_0.parquet".
    If the directory already exists, the dataset  will be saved with an incremented
    file name based on the highest existing file index.

    Args:
        data (pl.DataFrame): The dataset to be saved as a Parquet file.
        path (Path): The directory path where the Parquet file will be saved.
        base_filename (str): The base name for the Parquet file.

    Returns:
        None

    Examples:
        >>> dataset = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> os.listdir("data")
        ['dataset_0.parquet']
        >>> save_next_parquet(Path("data"), dataset)
        >>> os.listdir("data")
        ['dataset_0.parquet', 'dataset_1.parquet']

    """
    if not path.is_dir() or not list(path.glob(f"{base_filename}_*.parquet")):
        path.mkdir(exist_ok=True, parents=True)
        save_dataframe(data, path / f"{base_filename}_0.parquet", fmt="parquet")
    else:
        last_written_file_index = max(int(p.stem.split("_")[1]) for p in path.glob(f"{base_filename}_*.parquet"))
        save_dataframe(data, path / f"{base_filename}_{last_written_file_index + 1}.parquet", fmt="parquet")


#################
##  Utilities  ##
#################


def create_polars_reader(
    fmt: str = "parquet",
    lazy_frame: bool = False,
    batched: bool = False,
    schema_only: bool = False,
    metadata: bool = False,
) -> Callable[[str | Path | list[str] | list[Path]], pl.DataFrame]:
    """Retrieve the Polars reader for the specified file format.

    Args:
        fmt (str): The file format for which to retrieve the Polars reader.
        lazy_frame (bool): Whether to retrieve a lazy frame reader. Defaults to False.
        batched (bool): Whether to retrieve a batched reader. Defaults to False.
        schema_only (bool): Whether to retrieve a schema-only reader. Defaults
        metadata (bool): Whether to retrieve a metadata-only reader. Defaults

    Returns:
        Callable: The Polars reader for the specified file format.

    Raises:
        ValueError: If the specified file format is not supported.

    Examples:
        >>> reader = create_polars_reader("csv")
        >>> df = reader("data.csv")
        >>> print(df)
        >>> shape: (2, 2)
        >>> ┌─────┬─────┐
        >>> | a   | b   |
        >>> | --- | --- |
        >>> | 1   | 2   |
        >>> | 3   | 4   |
        >>> └─────┴─────┘

    """
    from polars.io import __all__

    # Determine the prefix based on whether a lazy frame is requested
    prefix = "scan" if lazy_frame else "read"

    # Filter the readers based on the prefix
    readers = [
        r
        for r in __all__
        if r.startswith(prefix)
        and ("batched" in r) == batched
        and ("schema" in r) == schema_only
        and ("metadata" in r) == metadata
    ]

    # Create a dictionary mapping file formats to their corresponding readers
    reader_dict = {reader.split("_")[1]: getattr(pl.io, reader) for reader in readers}

    # Retrieve the reader for the specified file format
    if fmt in reader_dict:
        return reader_dict[fmt]  # type: ignore[no-any-return]
    else:
        raise ValueError(f"Unsupported file format: {fmt}")


def infer_datetime_dtype_for_casting(
    schema: pl.Schema, column: str, *, time_zone: str | None = None, time_unit: Literal["ns", "us", "ms"] = "ns"
) -> pl.Expr:
    """Infer the datetime dtype for casting a specified column in a Polars DataFrame.

    Args:
        schema (pl.DataFrame): The schema of the DataFrame containing the column.
        column (str): The name of the column to be cast to datetime.
        time_zone (Optional[str]): The time zone to use for the datetime column. Defaults to None.
        time_unit (Optional[str]): The time unit for the datetime column. Defaults to "ns".

    Returns:
        pl.Expr: An expression representing the column cast to datetime.

    """
    if schema[column] == pl.String or schema[column] == pl.Utf8:
        cdate = pl.col(column).str.to_datetime(time_zone=time_zone)
    elif schema[column] == pl.Datetime or schema[column] == pl.Date:
        cdate = pl.col(column)
    else:
        raise PolarsError(f"Unsupported column type for datetime conversion: {schema[column]}")

    return cdate.cast(pl.Datetime(time_unit=time_unit, time_zone=time_zone))
