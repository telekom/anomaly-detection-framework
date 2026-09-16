# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import drain3.masking
import numpy as np
import polars as pl

from collections.abc import Callable, Collection, Iterable
from drain3 import TemplateMiner
from numpy.typing import NBitBase, NDArray
from pathlib import Path
from typing import Any, TypeVar, cast

from adf.core.dataframe.api import add_replaced_string
from adf.core.dataframe.base import load_dataframe

T = TypeVar("T", bound=NBitBase)


def _extract_masking(template_miner: TemplateMiner) -> list[tuple[str, list[drain3.masking.MaskingInstruction]]]:
    """Extract masking instructions from TemplateMiner.

    Args:
        template_miner (TemplateMiner): Template miner.

    Returns:
        rust_pattern (list[str]): List of regex patterns to extract masking to apply in Polars.

    """
    mi = template_miner.masker.mask_name_to_instructions
    return list(zip(mi.keys(), mi.values()))


def _check_clustering_metric(metric: str) -> Callable[..., np.number[T]]:
    """Check if the provided clustering metric is valid and returns the corresponding metric function.

    Args:
        metric (str): The name of the clustering metric to check.

    Returns:
        Callable: The clustering metric function from sklearn.metrics.cluster.

    Raises:
        ValueError: If the provided metric is not a valid clustering metric.

    """
    from sklearn.metrics.cluster import __all__

    if metric not in __all__:
        raise ValueError(f"Invalid metric '{metric}'. Available metrics are: {__all__}")

    return cast(Callable[..., np.number[T]], getattr(__import__("sklearn.metrics.cluster", fromlist=[metric]), metric))


def _as_polars_dataframe(
    X: str | Path | Iterable[Path] | Iterable[str] | NDArray[Any] | pl.DataFrame | pl.LazyFrame, **kwargs: Any
) -> pl.DataFrame | pl.LazyFrame:
    """Check the type of the input and convert it to a Polars DataFrame.

    Args:
    X : pl.DataFrame, str, Path, list, or np.ndarray
        The input data to be checked and converted.

    **kwargs : dict
        Additional keyword arguments to be passed to the `load_dataframe` function if `X` is a string or Path.

    Returns:
    pl.DataFrame
        The input data converted to a Polars DataFrame if necessary.

    Raises:
    TypeError
        If the input data type is not supported.

    """
    _supported_types = (pl.DataFrame, pl.LazyFrame, str, Path, list, np.ndarray)

    if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
        return X
    elif isinstance(X, (str, Path)):
        return load_dataframe(X, **kwargs)
    elif isinstance(X, list):
        if isinstance(X[0], Path):
            return load_dataframe(X, **kwargs)
        return pl.from_records(X)
    elif isinstance(X, np.ndarray):
        return pl.from_numpy(X)
    else:
        raise TypeError(f"Unsupported input type '{type(X)}'. Supported types are: {_supported_types}")


def _check_X(
    X: pl.DataFrame | pl.LazyFrame,
    pattern: Collection[tuple[str, Collection[drain3.masking.MaskingInstruction]]] | None = None,
    return_unique: bool = True,
    slice_length: int | None = None,
    rstrip: bool = True,
) -> list[str]:
    """Check all the necessary conditions for the input data.

    Args:
        X (pl.DataFrame): The input DataFrame which must have only one column.
        pattern (str, optional): The regex pattern to be used for string replacement. Defaults to None.
        return_unique (bool, optional): If True, returns a list of unique values from the column. Defaults to True.
        slice_length (int, optional): The length of the slice to extract. Defaults to None.
        rstrip (bool, optional): If True, strips leading and trailing whitespace from column. Defaults to True.

    Returns:
        list[str]: A list of values from the DataFrame column. If return_unique is True,
            the list contains unique values only, in order of first appearance.

    Raises:
        ValueError: If the input DataFrame does not have exactly one column.

    Note:
        Deduplication keeps the input order. Drain is order-dependent — the first message to
        reach a cluster creates it and later ones merge into it — so an unordered ``unique()``
        makes ``fit()`` non-deterministic: the same corpus fitted twice yields different
        templates and different cluster ids.

    """
    if len(X.collect_schema()) != 1:
        raise ValueError("The input DataFrame must have only one column.")

    col = X.collect_schema().names()[0]

    if X.collect_schema().dtypes()[0] != pl.String:
        raise ValueError("The input DataFrame column must be of type Utf8.")

    if return_unique:
        X = X.unique(maintain_order=True)

    if slice_length is not None:
        X = X.with_columns(pl.col(col).str.slice(offset=0, length=slice_length))

    if rstrip:
        X = X.with_columns(pl.col(col).str.strip_chars())

    if pattern is not None:
        for ma_pa in pattern:
            for pa in ma_pa[1]:
                X = add_replaced_string(X, column=col, pattern=pa.pattern, replacement=ma_pa[0], out_column=col)

    X = X.collect() if isinstance(X, pl.LazyFrame) else X
    return X.to_series().to_list()
