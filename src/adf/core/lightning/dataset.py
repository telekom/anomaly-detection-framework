# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import h5py
import numpy as np
import torch
import torch.nn as nn

from h5py import Dataset as H5Dataset
from numpy.typing import NDArray
from pathlib import Path
from torch import LongTensor, Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from typing import Any, Literal
from typing_extensions import Self, override


class WindowEmbeddingLookup(Dataset[Any]):
    """A class to handle embedding lookups for different backends."""

    embeddings: NDArray[np.float_] | nn.Embedding
    window_indexes: LongTensor | NDArray[np.int_] | H5Dataset

    def __init__(
        self,
        embeddings: nn.Embedding | NDArray[np.float_],
        window_indexes: LongTensor | NDArray[np.int_] | H5Dataset,
        backend: Literal["numpy", "torch", "numpy_hdf5"] = "numpy",
        return_tensor: bool = True,
        fastindex: bool = False,
        masked: bool = False,
    ):
        """Initialize the EmbeddingLookup instance.

        Args:
            embeddings (nn.Embedding | ndarray): The embedding matrix, either a numpy array or a PyTorch tensor.
            window_indexes (LongTensor | ndarray | H5Dataset): The indexes corresponding to the embedding windows,
                either a numpy array or a PyTorch tensor.
            backend (Literal["numpy", "torch", "numpy_hdf5"]): The backend to use for operations. Options are "numpy"
                or "torch". Defaults to "numpy".
            return_tensor (bool): If True, return the output as a PyTorch tensor. Defaults to True.
            fastindex (bool): If True, use a faster indexing method. Defaults to False.
            masked (bool): If True, creates a mask from embeddings that are zeros. Defaults to False.

        """
        self.embeddings = embeddings
        self.window_indexes = window_indexes
        self.backend = backend
        self.return_tensor = return_tensor
        self.masked = masked

        self.__assert_correct_types()
        if fastindex:
            self.__getitems__ = self.__getitems_impl__

    def __assert_correct_types(self) -> None:
        if self.backend not in ["numpy", "torch", "numpy_hdf5"]:
            raise NotImplementedError(f"Backend {self.backend} not implemented")

        if not isinstance(self.window_indexes, (np.ndarray, torch.Tensor, H5Dataset)) or not isinstance(
            self.embeddings, (np.ndarray | nn.Embedding)
        ):
            raise NotImplementedError(
                f"window_indexes and embeddings must be np.ndarray or torch.Tensor, but got {type(self.window_indexes)}"
                f"and {type(self.embeddings)}"
            )

        if isinstance(self.embeddings, np.ndarray) and not isinstance(self.window_indexes, (np.ndarray, H5Dataset)):
            raise NotImplementedError(
                f"window_indexes must be np.ndarray or H5Dataset, but got {type(self.window_indexes)}"
            )

        if isinstance(self.embeddings, nn.Embedding) and not isinstance(self.window_indexes, torch.Tensor):
            raise NotImplementedError(f"window_indexes must be torch.Tensor, but got {type(self.window_indexes)}")

        if self.backend == "numpy" and (
            not isinstance(self.embeddings, np.ndarray) or not isinstance(self.window_indexes, np.ndarray)
        ):
            raise NotImplementedError("embeddings and window_indexes must be np.ndarray when using numpy backend")

        if self.backend == "torch" and (
            not isinstance(self.window_indexes, torch.Tensor) or not isinstance(self.embeddings, nn.Embedding)
        ):
            raise NotImplementedError("embeddings must be nn.Embedding and window_indexes must be torch.Tensor")

        if self.backend == "numpy_hdf5" and (
            not isinstance(self.window_indexes, H5Dataset) or not isinstance(self.embeddings, np.ndarray)
        ):
            raise NotImplementedError("embeddings must be np.ndarray and window_indexes must be H5Dataset")

    def __getitems_impl__(self, idxs: list[int]) -> Tensor | NDArray[np.float_] | tuple[Any, ...]:
        """Retrieve the embeddings corresponding to the given indexes.

        This method fetches the embedding vector for the specified index
        from the embedding matrix, using the backend specified during initialization.

        Args:
            idxs (int): The index or indices of the embedding(s) to retrieve.

        Returns:
            numpy.ndarray or torch.Tensor: The embedding vector(s) corresponding to the given index.

        """
        if self.backend == "numpy" or self.backend == "numpy_hdf5":
            out = self.embeddings[self.window_indexes[idxs]]
            if self.return_tensor:
                out = torch.from_numpy(out)
        elif self.backend == "torch":
            out = self.embeddings(self.window_indexes[idxs])  # type: ignore
        else:
            raise NotImplementedError(f"Backend {self.backend} not implemented")

        if self.masked:
            if isinstance(out, torch.Tensor):
                mask = out.ne(0).any(dim=-1, keepdim=True).expand_as(out)
            else:
                mask = np.broadcast_to(np.any(out != 0, axis=-1, keepdims=True), out.shape)  # type: ignore
            return out, mask

        return out  # type: ignore

    @override
    def __getitem__(self, idx: int) -> Tensor | NDArray[np.float_] | tuple[Any, ...]:
        """Retrieve the embedding corresponding to the given index.

        This method fetches the embedding vector for the specified index
        from the embedding matrix, using the backend specified during initialization.

        Args:
            idx (int): The index or indices of the embedding(s) to retrieve.

        Returns:
            numpy.ndarray or torch.Tensor: The embedding vector(s) corresponding to the given index.

        """
        if self.backend == "numpy" or self.backend == "numpy_hdf5":
            out = self.embeddings[self.window_indexes[idx], :]
            if self.return_tensor:
                out = torch.from_numpy(out)
        elif self.backend == "torch":
            out = self.embeddings(self.window_indexes[idx])  # type: ignore
        else:
            raise NotImplementedError(f"Backend {self.backend} not implemented")

        if self.masked:
            if isinstance(out, torch.Tensor):
                mask = out.ne(0).any(dim=-1, keepdim=True).expand_as(out)
            else:
                mask = np.broadcast_to(np.any(out != 0, axis=-1, keepdims=True), out.shape)  # type: ignore
            return out, mask

        return out

    def __len__(self) -> int:
        """Return the number of elements in the `window_indexes`.

        This method determines the length of the `window_indexes` attribute, which can be either a numpy
        array or a sequence.

        Returns:
            length (int): The number of elements in `window_indexes`.

        """
        return self.window_indexes.shape[0]

    @classmethod
    def from_ndarray(cls, embeddings: NDArray[np.float_], window_indexes: NDArray[np.int_], **kwargs: Any) -> Self:
        """Create an EmbeddingLookup instance from numpy arrays.

        Args:
            embeddings (np.ndarray): The embedding matrix as a numpy array.
            window_indexes (np.ndarray): The indexes corresponding to the embedding windows as a numpy array.
            **kwargs: Additional keyword arguments for the EmbeddingLookup instance.

        Returns:
            WindowEmbeddingLookup: An instance of the EmbeddingLookup class initialized with the provided numpy arrays.

        Raises:
            ValueError: If `embeddings` or `window_indexes` are not numpy arrays.

        """
        if not isinstance(embeddings, np.ndarray) or not isinstance(window_indexes, np.ndarray):
            raise ValueError("embeddings and window_indexes must be np.ndarray")

        return cls(embeddings, window_indexes, backend="numpy", **kwargs)

    @classmethod
    def from_torch(cls, embeddings: torch.Tensor | nn.Embedding, window_indexes: torch.Tensor, **kwargs: Any) -> Self:
        """Create an EmbeddingLookup instance from PyTorch tensors.

        Args:
            embeddings (torch.Tensor): The embedding matrix as a PyTorch tensor.
            window_indexes (torch.Tensor): The indexes corresponding to the embedding windows as a PyTorch tensor.
            **kwargs: Additional keyword arguments for the EmbeddingLookup instance.

        Returns:
            WindowEmbeddingLookup: An instance of the EmbeddingLookup class initialized with the provided PyTorch
                tensors.

        Raises:
            ValueError: If `embeddings` or `window_indexes` are not PyTorch tensors.

        """
        if not isinstance(embeddings, (torch.Tensor, nn.Embedding)) or not isinstance(window_indexes, torch.Tensor):
            raise ValueError("embeddings and window_indexes must be torch.Tensor")

        if isinstance(embeddings, torch.Tensor):
            nn_emb = nn.Embedding.from_pretrained(embeddings)
            return cls(nn_emb, window_indexes, backend="torch", **kwargs)

        return cls(embeddings, window_indexes, backend="torch", **kwargs)

    @classmethod
    def from_h5py(cls, embeddings: NDArray[np.float_], window_indexes: str | Path | h5py.File, **kwargs: Any) -> Self:
        """Create an EmbeddingLookup instance from h5py files.

        Args:
            embeddings (NDArray[np.float_]): The embedding matrix as a numpy array.
            window_indexes (str | Path | h5py.File): The path to the HDF5 file or an h5py.File object
                containing the window indexes.
            **kwargs: Additional keyword arguments for the EmbeddingLookup instance.

        """
        if not isinstance(embeddings, np.ndarray) or not isinstance(window_indexes, (str, Path, h5py.File)):
            raise ValueError("embeddings must be numpy array and window_indexes must be str or Path or h5py.File")

        if isinstance(window_indexes, str | Path):
            window_indexes = h5py.File(window_indexes, "r")

        if len(window_indexes.keys()) != 1:
            raise ValueError("window_indexes must contain one virtual dataset")

        arr_key = list(window_indexes.keys())[0]
        if window_indexes[arr_key].ndim not in [2, 3]:
            raise ValueError(
                f"Dataset {arr_key} in window_indexes must be 2D or 3D, but got {window_indexes[arr_key].ndim}"
            )

        return cls(embeddings, window_indexes[arr_key], backend="numpy_hdf5", **kwargs)

    @classmethod
    def load(
        cls,
        embeddings_path_or_array: str | Path,
        window_indexes_path_or_array: str | Path,
        backend: Literal["numpy", "torch", "numpy_hdf5"] = "numpy_hdf5",
        **kwargs: Any,
    ) -> Self:
        """Load embeddings and window indexes from files and create an EmbeddingLookup instance.

        Args:
            embeddings_path_or_array (str | Path | ndarray | Tensor | Embedding): Path to the file containing the
                embedding matrix or the matrix itself.
            window_indexes_path_or_array (str | Path): Path to the file containing the window indexes.
            backend (Literal["numpy", "torch"], optional): The backend to use for loading and operations.
                Options are "numpy" or "torch". Defaults to "numpy".
            **kwargs: Additional keyword arguments for loading PyTorch tensors.

        Returns:
            WindowEmbeddingLookup: An instance of the EmbeddingLookup class initialized with the loaded data.

        Raises:
            ValueError: If `embeddings_path` or `window_indexes_path` are not strings.
            NotImplementedError: If the specified backend is not supported.

        """
        if not isinstance(embeddings_path_or_array, str | Path) or not isinstance(
            window_indexes_path_or_array, str | Path
        ):
            raise ValueError("embeddings_path and window_indexes_path must be strings or paths")

        if backend == "numpy":
            embeddings = (
                np.load(embeddings_path_or_array)
                if isinstance(embeddings_path_or_array, (str, Path))
                else embeddings_path_or_array
            )
            window_indexes = (
                np.load(window_indexes_path_or_array)
                if isinstance(window_indexes_path_or_array, (str, Path))
                else window_indexes_path_or_array
            )
            return cls.from_ndarray(embeddings, window_indexes, **kwargs)
        elif backend == "torch":
            embeddings = (
                torch.load(embeddings_path_or_array, weights_only=True)
                if isinstance(embeddings_path_or_array, (str, Path))
                else embeddings_path_or_array
            )
            window_indexes = (
                torch.load(window_indexes_path_or_array, weights_only=True)
                if isinstance(window_indexes_path_or_array, (str, Path))
                else window_indexes_path_or_array
            )
            return cls.from_torch(embeddings, window_indexes, **kwargs)
        elif backend == "numpy_hdf5":
            embeddings = (
                np.load(embeddings_path_or_array)
                if isinstance(embeddings_path_or_array, (str, Path))
                else embeddings_path_or_array
            )
            window_indexes = (
                h5py.File(str(window_indexes_path_or_array), "r")
                if isinstance(window_indexes_path_or_array, (str, Path))
                else window_indexes_path_or_array
            )
            return cls.from_h5py(embeddings, window_indexes, **kwargs)
        else:
            raise NotImplementedError(f"Backend {backend} not implemented")

    @override
    def __repr__(self) -> str:
        """Return a string representation of the EmbeddingLookup instance."""
        if self.backend == "numpy":
            return (
                f"EmbeddingLookup(embeddings=ndarray{self.embeddings.shape},"
                f"window_indexes=ndarray{self.window_indexes.shape} backend={self.backend})"
            )
        elif self.backend == "torch":
            shape = self.embeddings.weight.shape  # type: ignore
            return (
                f"EmbeddingLookup(embeddings=nn.Embedding{shape},"
                f"window_indexes=LongTensor{self.window_indexes.shape}, backend={self.backend})"
            )
        elif self.backend == "numpy_hdf5":
            return (
                f"EmbeddingLookup(embeddings=ndarray{self.embeddings.shape},"
                f"window_indexes=HDF5(virtual_dataset({self.window_indexes.shape}), backend={self.backend})"
            )
        else:
            raise NotImplementedError

    def as_dataloader(self, batch_size: int, **kwargs: Any) -> DataLoader[Tensor]:
        """Create a DataLoader from the dataset.

        Args:
            batch_size (int): The size of each batch.
            **kwargs: Additional keyword arguments for DataLoader.

        Returns:
            DataLoader: A DataLoader instance for the dataset.

        """
        return DataLoader(self, batch_size=batch_size, **kwargs)
