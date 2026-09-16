# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import polars as pl

from datetime import date, timedelta
from polars._typing import FrameInitTypes
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from typing import Any
from typing_extensions import override

from adf.core.dataframe.api import add_forward_sliding_window


class ArrowRecordsDataset(Dataset[Any]):
    """Polars dataset for ADF inference.

    Attributes:
        dataset (pl.DataFrame): The dataset containing the records.

    """

    def __init__(self, dataset: FrameInitTypes, column: str, datetime_column: str, window_size: int):
        """Initialize the dataset with the specified records.

        Args:
            dataset (FrameInitTypes): The dataset containing the records.
            column (str): The column of logs to create windows from.
            datetime_column (str): The column of dates.
            window_size (int): The size of the window.

        """
        if window_size < 1 or not isinstance(window_size, int):
            raise ValueError("Window size must be an integer greater than 0")

        self.dataset = self.generate_sliding_windows(
            dataset, column=column, datetime_column=datetime_column, window_size=window_size
        )

        if not isinstance(self.dataset.schema[datetime_column], pl.Datetime):
            raise pl.exceptions.SchemaError(f"Column {datetime_column} must be of type Date")

    def __len__(self) -> int:
        """Return the number of elements in the dataset.

        Returns:
            len int: The number of elements in the dataset.

        """
        return len(self.dataset)

    @override
    def __getitem__(self, idx: int) -> tuple[list[str], timedelta | date | Any]:
        """Retrieve an item from the dataset by its index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict[str, Any]: The dataset item corresponding to the given index.

        """
        return self.dataset.row(idx)[0], [self.dataset.row(idx)[1]]

    @staticmethod
    def collate_fn(batch: list[tuple[list[str], timedelta | date | Any]]) -> tuple[Tensor, Tensor]:
        """Collate the batch of items into a list.

        Args:
            batch (list[tuple[list[str], timedelta | date | Any]]): The batch of items to collate.

        Returns:
            tuple[Tensor, Tensor]: The collated batch of items.

        """
        logs, dates = zip(*batch)
        logs = [log for log in logs]
        dates = [date for date in dates]

        return logs, dates

    def as_dataloader(self, batch_size: int, **kwargs: Any) -> DataLoader[Tensor]:
        """Create a DataLoader from the dataset.

        Args:
            batch_size (int): The batch size for the DataLoader.
            **kwargs (dict[str, Any]): Additional keyword arguments for the DataLoader.

        Returns:
            DataLoader: The DataLoader created from the dataset.

        """
        return DataLoader(self, batch_size=batch_size, collate_fn=self.collate_fn, **kwargs)

    @staticmethod
    def generate_sliding_windows(
        dataset: FrameInitTypes, column: str, datetime_column: str, window_size: int
    ) -> pl.DataFrame:
        """Create a sliding window view of the specified column in the dataset.

        Args:
            dataset (FrameInitTypes): The input data to be converted into a DataFrame.
            column (str): The name of the logs column to apply the sliding window on.
            datetime_column (str): The name of the datetime column.
            window_size (int): The size of the sliding window.

        Returns:
            pl.DataFrame: A DataFrame with the sliding window applied to the specified column.

        Raises:
            ValueError: If the input data type is not supported.

        """
        try:
            dataset = pl.DataFrame(dataset)
        except Exception as e:
            raise ValueError(f"Not supported data type: {type(dataset)}") from e

        dataset = add_forward_sliding_window(
            dataset, column=column, window_size=window_size, output=f"{column}_window", filter_valid_rows=True
        )

        if dataset.is_empty():
            raise pl.exceptions.NoDataError("No rows in the dataset")

        return dataset.select([f"{column}_window", datetime_column])
