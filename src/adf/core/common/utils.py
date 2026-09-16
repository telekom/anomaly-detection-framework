# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import numpy as np
import os
import random

from collections.abc import Sequence
from numpy.typing import DTypeLike
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def make_deterministic(seed: int = 42) -> None:
    """Set the random seed for various libraries to ensure deterministic behavior.

    This function sets the seed for Python's built-in random module, NumPy, and PyTorch
    (both CPU and GPU, if available). It also configures PyTorch to use deterministic
    algorithms and disables the benchmarking feature to ensure reproducibility.

    Args:
        seed (int, optional): The seed value to use for random number generation.
                              Defaults to 42.

    Returns:
        None

    """
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def assert_memmap_tensor_dtype(np_dtype: DTypeLike, tensor_dtype: torch.dtype) -> None:
    """Assert that the data type of a memory-mapped NumPy array matches the data type of a PyTorch tensor.

    Args:
        np_dtype (numpy.dtype): The data type of the memory-mapped NumPy array.
        tensor_dtype (torch.dtype): The data type of the PyTorch tensor.

    Raises:
        ValueError: If the `np_dtype` is not supported or if the `tensor_dtype` does not match the `np_dtype`.

    Examples:
        >>> assert_memmap_tensor_dtype(np.float32, torch.float32)
        >>> assert_memmap_tensor_dtype(np.float32, torch.float64)
        ValueError: Tensor dtype torch.float64 does not match memmap dtype np.float32

    """
    import torch

    numpy_to_torch_dtype_dict: dict[DTypeLike, torch.dtype] = {
        np.bool_: torch.bool,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,
    }

    if np_dtype not in numpy_to_torch_dtype_dict:
        raise ValueError(f"Unsupported dtype {np_dtype}")

    if numpy_to_torch_dtype_dict[np_dtype] != tensor_dtype:
        raise ValueError(f"Tensor dtype {tensor_dtype} does not match memmap dtype {np_dtype}")


def get_device_accelerator(device: str | None = None) -> torch.device:
    """Determine the available device accelerator and return its type and torch device.

    This function checks for the availability of CUDA and MPS (Metal Performance Shaders) accelerators.
    If neither is available, it defaults to the CPU.

    Args:
        device (str | None, optional): The specific device to use.
            If None, the current CUDA device is used if available. Defaults to None.

    Returns:
        device (torch.device): The torch device is the corresponding PyTorch device for the selected accelerator.

    """
    import torch

    if torch.cuda.is_available():
        accel = "cuda"
        torch_device = (
            torch.device(f"{accel}:{torch.cuda.current_device()}") if device is None else torch.device(f"cuda:{device}")
        )
        logging.info(f"Using CUDA device: {torch_device}")
    elif torch.backends.mps.is_available():
        if device is not None:
            raise RuntimeError("MPS device does not support specifying a device ID.")
        accel = "mps"
        torch_device = torch.device(accel)
        logging.info("Using MPS device")
    else:
        accel = "cpu"
        if device is not None:
            raise RuntimeError("CPU device does not support specifying a device ID.")
        torch_device = torch.device(accel)
        logging.info("Using CPU device")

    return torch_device


def format_requirements(requirements: Sequence[str] | None) -> list[str]:
    """Expand a sequence of requirements list.

    This function processes each item in the input sequence. If an item is a string ending with `.txt`,
    it treats it as a requirements file, reads its contents (ignoring empty lines and comments), and adds each valid
    requirement to the result. Otherwise, the item is added directly.

    Args:
        requirements (Sequence[str]): A sequence of requirement strings or paths to .txt files containing requirements.

    Returns:
        Sequence[str]: A flat list of requirement strings.

    Raisess:
        FileNotFoundError: If a specified .txt file does not exist.

    Examples:
        >>> format_requirements(["requirements.txt", "torch>=2.0"])
        ['numpy>=1.21.0', 'torch>=2.0']

    """
    result = []
    if requirements:
        for req in requirements:
            if isinstance(req, str) and req.endswith(".txt"):
                if not os.path.isfile(req):
                    raise FileNotFoundError("Requirements file not found")
                with open(req) as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                    result.extend(lines)
            else:
                result.append(req)
    return result
