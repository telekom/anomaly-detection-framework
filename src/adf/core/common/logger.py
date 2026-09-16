# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time

from collections.abc import Callable, Collection
from logging import Logger
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def setup_logging(filename: str, console_level: str = "INFO") -> None:
    """Configure logging with a file handler (DEBUG level) and a stdout handler (configurable level).

    Args:
        filename (str): Path of the file to write logs to.
        console_level (str): Log level name for console output (e.g. "DEBUG", "INFO", "WARNING").
            Defaults to ``"INFO"``.

    The log format includes a timestamp, log level, and message. The root logger captures messages at DEBUG level.

    """
    level = getattr(logging, console_level.upper(), logging.INFO)

    # Create a file handler for logging to a file with DEBUG level
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler.setFormatter(formatter)

    # Create a logging handler for tqdm (writes directly to log without interrupting the progress bar)
    tqdm_handler = logging.StreamHandler()
    tqdm_handler.setLevel(level)
    tqdm_handler.setFormatter(formatter)

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Root logger should capture DEBUG messages
    root_logger.addHandler(file_handler)  # Logs to file at DEBUG level
    root_logger.addHandler(tqdm_handler)  # Logs to console at configured level

    # Suppress Lightning's verbose output unless DEBUG is requested
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING if level > logging.DEBUG else level)
    # Quiet noisy AWS client loggers
    if level > logging.DEBUG:
        for name in ("boto3", "botocore", "s3transfer", "urllib3", "asyncio"):
            logging.getLogger(name).setLevel(logging.WARNING)


def log_total_system_disk_usage(logger: Logger, where: str) -> None:
    """Log total, used, and free disk space for the first partition exceeding 1GB and matching specific criteria.

    Also logs current and maximum RAM usage. Skips inaccessible partitions.

    Args:
        logger (Logger): Logger object to use for logging.
        where (str): Identifier to include in log messages (e.g., process or location description).

    """
    import psutil

    for partition in psutil.disk_partitions(all=True):
        try:
            # Get disk usage details for each partition mount point
            usage = psutil.disk_usage(partition.mountpoint)
            total_gb = usage.total / (1024**3)
            used_gb = usage.used / (1024**3)
            free_gb = usage.free / (1024**3)

            # Log details for partitions larger than 1GB and matching the device criteria
            if used_gb > 1.0 and partition.device.startswith("/dev/nvme1"):
                logger.info("%s --- Mount Point: %s", where, partition.mountpoint)
                logger.info(
                    "  Device: %s | Total: %.2f GB | Used: %.2f GB | Free: %.2f GB",
                    partition.device,
                    total_gb,
                    used_gb,
                    free_gb,
                )
                break
        except PermissionError:
            # Skip inaccessible mount points
            logger.warning("Mount Point: %s - Access Denied", partition.mountpoint)
    logger.info("-" * 40)


def log_files_size(logger: Logger, where: str, files: Collection[Path], mem_type: Literal["MB", "GB"] = "GB") -> None:
    """Log the total size of a list of files in gigabytes.

    Args:
        logger (Logger): Logger object to use for logging.
        where (str): A description of where the files are located or any relevant context.
        files (list[Path]): A list of file paths whose sizes are to be summed and logged.
        mem_type (Literal["MB", "GB"], optional): The unit of measurement. Can be "MB" or "GB". Defaults to "GB".

    Returns:
        None

    """
    if mem_type == "GB":
        norm = 1024**3
    elif mem_type == "MB":
        norm = 1024**2
    else:
        raise ValueError(f"Invalid memory type: {mem_type}")

    total_size_bytes = sum(file.stat().st_size for file in files)
    total_size_gb = total_size_bytes / norm
    logger.info("%s Total size of files: %.5f (%s)", where, total_size_gb, mem_type)


def timelog(func: F) -> F:
    """Measure and prints the execution time of the decorated function.

    Args:
        func (Callable[..., Any]): The function to be decorated.

    Returns:
        result: The result of the decorated function.

    The decorator logs the name of the function and the time it took to execute in seconds.

    """

    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def] # noqa: ANN202, ANN002, ANN003
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time: float = time.perf_counter()
        execution_time = end_time - start_time
        logger.info("%s executed in [%.4f] seconds", func.__name__, execution_time)
        return result

    return cast(F, wrapper)
