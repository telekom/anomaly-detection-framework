# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import lightning as L
import multiprocessing as mp
import numpy as np
import torch.nn as nn

from collections.abc import Mapping
from numpy.typing import NDArray
from pathlib import Path
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import Any, Literal, Union, cast
from typing_extensions import TypeAlias, override

from adf.core.lightning.dataset import WindowEmbeddingLookup

from .utils import default_collate_fastindex

EmbeddingType: TypeAlias = Union[str, Path, Tensor, NDArray[np.float_], nn.Embedding]
WindowIndexType: TypeAlias = Union[Path, Tensor, NDArray[np.int_]]


class LogDataModule(L.LightningDataModule):
    """Lightning DataModule for loading and processing log data."""

    train_dataset: Dataset[Any]
    val_dataset: Dataset[Any] | None
    test_dataset: Dataset[Any] | None

    def __init__(
        self,
        embeddings: EmbeddingType,
        windows: WindowIndexType | Mapping[str, WindowIndexType],
        backend: Literal["numpy", "torch", "numpy_hdf5"] = "numpy_hdf5",
        fastindex: bool = True,
        masked: bool = False,
        sampler: str | None = None,
        sampler_kwargs: Mapping[str, Any] | None = None,
        dataloader_kwargs: Mapping[str, Any] | None = None,
    ):
        """Initialize the LogDataModule.

        Args:
            embeddings (str | Path | Tensor | np.ndarray): The embedding matrix.
            windows (str | Path | Tensor | np.ndarray): The indexes corresponding to the embedding windows.
            sampler (str): The name of the sampler.
            sampler_kwargs (dict): Additional arguments for the sampler.
            backend (Literal["numpy", "torch", "numpy_hdf5"]): The backend to use for loading data.
                Default is "numpy_hdf5".
            fastindex (bool): Whether to use fast indexing for the dataset. Default is True.
            masked (bool): If True, creates a tuple with tensors with one that contains the  mask from embeddings
                that are zeros. Defaults to False.
            dataloader_kwargs (dict): Additional arguments for the dataloader.

        """
        super().__init__()
        self.embeddings = embeddings
        self.windows = windows
        self.fastindex = fastindex
        self.masked = masked
        self.backend = backend
        self.dataloader_kwargs = dataloader_kwargs or {}
        self.sampler_kwargs = sampler_kwargs or {}

        if (self.dataloader_kwargs.get("num_workers", 0) > 1) and self.backend == "numpy_hdf5":
            # h5py doesn't work with spawn
            mp.set_start_method("fork", force=True)

        if not isinstance(self.embeddings, (str, Path, Tensor, np.ndarray)):
            raise ValueError(f"Unsupported embeddings type: {type(self.embeddings)}")

        if isinstance(windows, Mapping) and ("train" not in windows or "val" not in windows):
            raise ValueError("Windows must contain at least 'train' and 'val' keys when using a mapping.")

        self.sampler = self.set_sampler(sampler, **self.sampler_kwargs)

        if self.fastindex:
            self.dataloader_kwargs.setdefault("collate_fn", default_collate_fastindex)

    @override
    def setup(self, stage: str) -> None:
        """Set up the data module for training, validation, or testing.

        Args:
            stage (str): The stage of the data module (train, val, test).

        """
        if stage == "fit":
            if isinstance(self.windows, Mapping):
                self.train_dataset = self._load_dataset(
                    self.embeddings,
                    self.windows["train"],
                    backend=self.backend,
                    fastindex=self.fastindex,
                    masked=self.masked,
                )
                # Use train dataset embeddings which are already loaded and in memory (embeddings are shared)
                self.val_dataset = self._load_dataset(
                    self.train_dataset.embeddings,
                    self.windows["val"],
                    backend=self.backend,
                    fastindex=self.fastindex,
                    masked=self.masked,
                )
            else:
                self.train_dataset = self._load_dataset(
                    self.embeddings, self.windows, backend=self.backend, fastindex=self.fastindex, masked=self.masked
                )
                self.val_dataset = None
        else:
            if isinstance(self.windows, Mapping):
                self.test_dataset = self._load_dataset(
                    self.embeddings,
                    self.windows["test"],
                    backend=self.backend,
                    fastindex=self.fastindex,
                    masked=self.masked,
                )
            else:
                self.test_dataset = self._load_dataset(
                    self.embeddings, self.windows, backend=self.backend, fastindex=self.fastindex, masked=self.masked
                )

    @staticmethod
    def _load_dataset(
        embeddings: EmbeddingType,
        windows: WindowIndexType,
        backend: Literal["numpy", "torch", "numpy_hdf5"] = "numpy_hdf5",
        **kwargs: Any,
    ) -> WindowEmbeddingLookup:
        """Load a dataset using the provided embeddings and window indices.

        This method selects the appropriate loading mechanism based on the types of the arguments.
        If both embeddings and windows are file paths (str or Path), the dataset is loaded from files.
        If both are torch Tensors, the dataset is loaded using a torch-specific method.
        If both are NumPy arrays, the dataset is loaded using a NumPy-specific method.

        Args:
            embeddings (EmbeddingType):
                The embedding data, which can be a file path (str or Path), torch Tensor, or NumPy array.
            windows (WindowIndexType):
                The window indices, which can be a file path (str or Path), torch Tensor, or NumPy array.
            backend (Literal["numpy", "torch", "numpy_hdf5"], optional):
                The backend to use for loading. Defaults to "numpy_hdf5".
            **kwargs (Any): Additional keyword arguments to pass to the dataset loader.

        Returns:
            WindowEmbeddingLookup: An instance encapsulating the loaded embedding and window data.

        Raises:
            ValueError:
                If embeddings and windows types do not match any supported combination.

        """
        if isinstance(embeddings, (str, Path)) and isinstance(windows, (str, Path)):
            return WindowEmbeddingLookup.load(embeddings, windows, backend=backend, **kwargs)
        elif isinstance(embeddings, Tensor | nn.Embedding) and isinstance(windows, Tensor):
            return WindowEmbeddingLookup.from_torch(embeddings, windows, **kwargs)
        elif isinstance(embeddings, np.ndarray) and isinstance(windows, np.ndarray):
            return WindowEmbeddingLookup.from_ndarray(embeddings, windows, **kwargs)
        else:
            raise TypeError(
                f"Unsupported types for embeddings and windows: {type(embeddings)}, {type(windows)}. "
                "Expected str, Path, Tensor, or np.ndarray."
            )

    def on_train_dataloader_fetch(self) -> None:
        """Call when the train dataloader is fetched."""
        pass

    def on_val_dataloader_fetch(self) -> None:
        """Call when the val dataloader is fetched."""
        pass

    def on_test_dataloader_fetch(self) -> None:
        """Call when the test dataloader is fetched."""
        pass

    @override
    def train_dataloader(self) -> Any:
        """Create the training dataloader.

        Returns:
            dataloader (DataLoader): The training dataloader.

        """
        self.on_train_dataloader_fetch()
        return DataLoader(self.train_dataset, sampler=self.sampler, **self.dataloader_kwargs)

    @override
    def val_dataloader(self) -> Any:
        """Create the validation dataloader.

        Returns:
            dataloader (DataLoader): The validation dataloader.

        """
        self.on_val_dataloader_fetch()
        val_dataloader = DataLoader(self.val_dataset, **self.dataloader_kwargs) if self.val_dataset is not None else []
        return val_dataloader

    @override
    def test_dataloader(self) -> Any:
        """Create the test dataloader.

        Returns:
            dataloader (DataLoader): The test dataloader.

        """
        self.on_test_dataloader_fetch()
        test_dataloader = (
            DataLoader(self.test_dataset, **self.dataloader_kwargs) if self.test_dataset is not None else []
        )
        return test_dataloader

    @staticmethod
    def set_sampler(sampler: str | None = None, **kwargs: Any) -> Sampler[int] | None:
        """Get the sampler for the data module.

        Args:
            sampler (str): The name of the sampler.
            **kwargs: Additional arguments for the sampler.

        Returns:
            Any: The sampler instance.

        """
        if sampler is None:
            return None

        if not isinstance(sampler, str):
            raise ValueError(f"Unsupported sampler type: {type(sampler)}")

        sampler_class = cast(
            Sampler[int], getattr(__import__("torch.utils.data", fromlist=[sampler]), sampler)(**kwargs)
        )
        return sampler_class
