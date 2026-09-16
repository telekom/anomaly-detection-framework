# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import tempfile

from collections.abc import Iterable
from functools import partial
from numpy.typing import ArrayLike, NDArray
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    import h5py

    from h5py import Dataset as H5Dataset

T = TypeVar("T")


def sliding_window(
    arr: ArrayLike, window_size: int, force_padding_with_values: bool = False, padding_constant: float = 0.0
) -> NDArray[Any]:
    """Create a sliding window view of the input array.

    Args:
        arr (ArrayLike): Input array to create windows from.
        window_size (int): Size of each window.
        force_padding_with_values (bool, optional): Whether to force padding of the window if window_size is smaller
            than its needed from the array. Defaults to False.
        padding_constant (float, optional): Padding constant. Defaults to 0.0

    Returns:
        NDArray: A view of the input array with the specified window size.

    Examples:
        >>> sliding_window([1, 2, 3, 4, 5], 3)
        array([[1, 2, 3],
               [2, 3, 4],
               [3, 4, 5]])

    """
    arr = np.asarray(arr)

    if arr.ndim != 1:
        raise TypeError("array must be a 1D array")

    if force_padding_with_values and arr.shape[0] < window_size:
        arr = np.pad(arr, (0, window_size - arr.shape[0]), "constant", constant_values=padding_constant)

    return np.lib.stride_tricks.sliding_window_view(arr, window_size)


def ndarray_to_hdf5(
    arr: NDArray[Any],
    filename: str | Path,
    name: str = "arr",
    compression: str = "gzip",
    chunks: tuple[int, int] | None = None,
) -> None:
    """Save a NumPy array to an HDF5 file.

    Args:
        arr (NDArray[Any]): The NumPy array to save.
        filename (str | Path): The path to the HDF5 file to create.
        name (str, optional): The name of the dataset in the HDF5 file. Defaults to "arr".
        compression (str, optional): The compression algorithm to use. Defaults to "gzip".
        chunks (tuple[int, int], optional): The chunk size for the dataset. Defaults to None.

    """
    import h5py

    with h5py.File(filename, "w", libver="latest") as f:
        f.create_dataset(name=name, data=arr, chunks=chunks, compression=compression)


def create_virtual_dataset(
    arr: Iterable[H5Dataset] | Path,
    layout_dtype: str,
    name: str = "arr",
    out: str | Path | None = None,
    fillvalue: Any = None,
    return_open: bool = True,
) -> None | h5py.File:
    """Create a virtual dataset from NumPy arrays or `.npy` files in a directory.

    Args:
        arr (Iterable[H5Dataset] | Path): The data to create a virtual dataset from. Can be an Iterable of H5Dataset
            instances or a Path to a directory containing `.h5` files.
        out (str | Path): The path to the output HDF5 file where the virtual dataset will be saved.
        layout_dtype (str): The data type for the virtual layout.
        name (str, optional): The name of the dataset in the virtual layout. Defaults to "arr".
        fillvalue (Any, optional): The fill value for missing entries in the dataset. Defaults to None.
        return_open (bool, optional): Whether to return an opened file. Defaults to True.

    Returns:
        None

    """
    import h5py

    from h5py import Dataset as H5Dataset

    out = out or f"{tempfile.gettempdir()}/virtual.h5"

    if not isinstance(arr, (Iterable, Path)):
        raise TypeError("arr must be an Iterable of H5Dataset or a Path to directory containing .npy files.")

    if isinstance(arr, Iterable):
        if not all(isinstance(a, H5Dataset) for a in arr):
            raise TypeError("All elements in the Iterable must be H5Dataset instances.")
        if not all(name in a for a in arr):
            raise ValueError(f"All datasets must contain the key '{name}'.")
    else:
        if not arr.is_dir():
            raise NotADirectoryError(f"{arr} is not a directory")
        if not any(f.suffix == ".h5" for f in arr.glob("*.h5")):
            raise ValueError(f"No .h5 files found in {arr}")

    if isinstance(arr, Iterable):
        shapes = [a[name].shape for a in arr]
    elif isinstance(arr, Path):
        shapes = [h5py.File(f)[name].shape for f in sorted(arr.glob("*.h5")) if f.is_file() and f.suffix == ".h5"]
    else:
        raise TypeError("arr must be an NDArray, Iterable of NDArray, or Path to directory containing .npy files.")

    second_dim = set(s[1] for s in shapes)
    if len(second_dim) != 1:
        raise ValueError("All arrays must have the same second dimension size for virtual datasets.")

    shape = (sum(s[0] for s in shapes), second_dim.pop())
    layout = h5py.VirtualLayout(shape=shape, dtype=layout_dtype)

    m_start = 0
    if isinstance(arr, Iterable):
        for a in arr:
            m_end = m_start + a.shape[0]
            vsource = h5py.VirtualSource(a, shape=a.shape, dtype=layout_dtype)
            layout[m_start:m_end, :] = vsource
            m_start = m_end
    else:
        for f in sorted(arr.glob("*.h5")):
            h5 = h5py.File(f, "r")[name]
            m_end = m_start + h5.shape[0]
            vsource = h5py.VirtualSource(h5)
            layout[m_start:m_end, :] = vsource
            m_start = m_end

    with h5py.File(out, "w", libver="latest") as f:
        f.create_virtual_dataset(name, layout, fillvalue=fillvalue)

    if return_open:
        dataset = h5py.File(out, "r")[name]
        dataset.source = partial(_inverse_virtual_getitem_, dataset, key=name)
        return dataset
    return None


def _inverse_virtual_getitem_(dataset: H5Dataset, index: int, key: str = "arr") -> tuple[int, str]:
    """Retrieve the original index of the virtual source from the virtual dataset.

    Args:
        dataset (H5Dataset): The virtual dataset from which to retrieve the original source.
        index (int): The index in the virtual dataset for which to find the original source.
        key (str, optional): The key of the virtual dataset. Defaults to "arr".

    Returns:
        tuple[int, str]: A tuple containing the index of the virtual source and its name.

    """
    if not dataset.is_virtual:
        raise ValueError("The dataset is not a virtual dataset.")

    if not isinstance(index, int):
        raise TypeError("Index must be an integer.")

    import h5py

    virtual_sources = dataset.virtual_sources()
    shapes = [h5py.File(src.file_name)[key].shape for src in virtual_sources]

    if index > dataset.shape[0] or index < 0:
        raise IndexError(f"Index {index} is out of bounds for the virtual dataset with shape {dataset.shape}.")

    for source, shape in zip(virtual_sources, shapes):
        if index < shape[0]:
            return index, source.file_name
        index -= shape[0]

    return -1, ""  # If not found, return -1 and an empty string
