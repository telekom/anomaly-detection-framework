# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import torch

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from typing import Any
from typing_extensions import override


class MetricsInferenceDataset(Dataset[torch.Tensor]):
    """Dataset for metrics inference."""

    def __init__(self, data: Tensor, tensor_dtype: torch.dtype = torch.float32):
        """Initialize the dataset with preprocessed metrics data.

        Args:
            data: Tensor of shape [num_windows, window_size, features]
            tensor_dtype: Data type for tensors

        """
        self.data = data.type(tensor_dtype)

    def __len__(self) -> int:
        """Return the number of windows in the dataset."""
        return len(self.data)

    @override
    def __getitem__(self, idx: int) -> Tensor:
        """Retrieve a single window of metrics data.

        Args:
            idx: Index of the window to retrieve

        Returns:
            Window of metrics data

        """
        return self.data[idx]

    def as_dataloader(self, batch_size: int, **kwargs: Any) -> DataLoader[torch.Tensor]:
        """Create a DataLoader from the dataset.

        Args:
            batch_size: Batch size for the DataLoader
            **kwargs: Additional arguments for DataLoader

        Returns:
            Configured DataLoader

        """
        return DataLoader(self, batch_size=batch_size, **kwargs)
