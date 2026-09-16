# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import logging
import os
import requests
import shortuuid
import shutil
import tarfile
import tempfile

from collections.abc import Callable, Iterable
from glob import glob
from itertools import chain
from pathlib import Path
from tqdm import tqdm


def uuid_name() -> str:
    """Generate a random date string concatenated with UUID (Universally Unique Identifier) string.

    This function creates a random UUID using 4 bytes of random data
    and returns it as a string.

    Returns:
        uuid_name (str): A string representation of currante date and a randomly generated UUID.

    """
    today = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    uuid_prefix = shortuuid.uuid()

    return f"{today}_{uuid_prefix}"


def get_relative_path(sciprt_abs_path: str) -> str:
    """Calculate the relative path of the currently executing script with respect to the current working directory.

    Args:
        sciprt_abs_path (str): The absolute path of the sciprt executable.

    Returns:
        str: The relative path of the script.

    """
    current_working_dir = os.getcwd()
    relative_path = os.path.relpath(sciprt_abs_path, start=current_working_dir)

    return relative_path


def to_path_object(path: str | Path) -> Path:
    """Convert a string path to a Path object.

    Args:
        path (str | Path): A string path or a Path object.

    Returns:
        Path: The pathlib.Path representation of the given path.

    """
    if isinstance(path, str):
        return Path(path)
    elif isinstance(path, Path):
        return path
    else:
        raise TypeError("The 'path' argument must be a string or a Path object.")


def create_files(filepath: str | list[str], exist_ok: bool = True) -> None:
    """Create file(s) if they do not exist.

    Args:
        filepath (str) | list(str): The path to the file to be created.
        exist_ok (bool): If True, the function will not raise an error if the file already exists.

    Returns:
        None

    """
    if not isinstance(filepath, (str, Path, list)):
        raise ValueError("filepath must be a string, a Path or a list of strings or paths.")

    def _create_file(fp: str) -> None:
        if not os.path.isfile(fp):
            with open(fp, "a+"):
                pass
        else:
            if not exist_ok:
                raise FileExistsError(f"File already exists: {fp}")

    if isinstance(filepath, (str, Path)):
        _create_file(filepath)
    else:
        for path in filepath:
            _create_file(path)


def copy_tree(source: str | Path, destination: str | Path) -> None:
    """Copy a file or directory to a new location.

    Args:
        source (str | Path): The source file or directory to copy from.
        destination (str | Path): The destination file or directory to copy to.

    Raises:
        FileNotFoundError: If the source file or directory does not exist.
        Exception: If an error occurs during the copy operation.

    """
    source = to_path_object(source)
    destination = to_path_object(destination)

    if not source.exists():
        raise FileNotFoundError(f"Source '{source}' does not exist.")

    try:
        shutil.copytree(source, destination)
    except Exception as e:
        raise e


def copy_contents_without_perms(source: str | Path, destination: str | Path) -> None:
    """Copy the contents of a directory to another directory without copying permissions.

    This function copies the contents of the source directory to the destination directory,
    but does not copy the source directory itself and does not preserve permissions.

    Args:
        source (str | Path): The source directory to copy from.
        destination (str | Path): The destination directory to copy to.

    Raises:
        FileNotFoundError: If the source or destination directory does not exist.
        NotADirectoryError: If the source or destination is not a directory.
        Exception: If an error occurs during the copy operation.

    """
    source = to_path_object(source)
    destination = to_path_object(destination)

    if not source.exists() or not destination.exists():
        raise FileNotFoundError(f"Source '{source}' or Destination '{destination}' does not exist")
    if not source.is_dir() or not destination.is_dir():
        raise NotADirectoryError(f"Source '{source}' or Destination '{destination}' is not a directory")

    try:
        for item in source.iterdir():  #
            target = destination / item.name
            if item.is_dir():
                # Copy the directory without copying permissions
                shutil.copytree(item, target, copy_function=shutil.copyfile, dirs_exist_ok=True)
            else:
                shutil.copyfile(item, target)

    except Exception as e:
        raise e


def create_dirs(*args: Iterable[str | Path], parents: bool = True, exist_ok: bool = True) -> None:
    """Create directories for multiple paths.

    Args:
        args (Iterable[str | Path]): A list of paths for which directories will be created.
        parents (bool): If True, the function will create parent directories as needed.
        exist_ok (bool): If True, the function will not raise an error if the directory already exists.

    Returns:
        None

    """
    for path in args:
        if not isinstance(path, (Path, str)):
            raise ValueError(f"Path must be a pathlib.Path or a string. Received {path}")

        try:
            if isinstance(path, Path):
                path.mkdir(parents=parents, exist_ok=exist_ok)
            else:
                os.makedirs(path, exist_ok=exist_ok)
        except Exception as e:
            raise e


def replace_dir(source: str | Path, destination: str | Path) -> None:
    """Replace a folder at `destination` with the folder at `source`, including all contents.

    Args:
        source (str | Path): Path to the source folder that will replace the destination.
        destination (str | Path): Path to the destination folder to be replaced.

    """
    source = to_path_object(source)
    destination = to_path_object(destination)

    # Check if the source folder exists and is a directory
    if not source.is_dir():
        raise NotADirectoryError(f"Source folder '{source}' does not exist or is not a directory.")

    # Remove the existing destination folder if it exists
    if destination.exists():
        remove_object(destination)

    # Copy the source folder to the destination
    try:
        shutil.copytree(source, destination)
    except shutil.Error as e:
        raise OSError("Failed to copy source folder to destination") from e


def extract_tar_file(tar_path: str | Path, extract_path: str | Path | None = None) -> Path:
    """Extracts a tar file to a specified directory.

    Args:
        tar_path (str): The path to the tar file to be extracted.
        extract_path (str, optional): The directory to extract the files to.
                                     If None, a temporary directory will be used.

    """
    if not Path(tar_path).exists():
        logging.error(f"Error: The file '{tar_path}' does not exist.")
        raise FileNotFoundError(f"Error: The file '{tar_path}' does not exist.")

    # Use a temporary directory if no extraction path is provided
    if extract_path is None:
        extract_path = tempfile.gettempdir()
        logging.info(f"No extraction path specified. Using temporary directory: {extract_path}")
    else:
        os.makedirs(extract_path, exist_ok=True)

    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_path)
        logging.info(f"Successfully extracted '{tar_path}' to '{extract_path}'.")
        return Path(extract_path)
    except Exception as e:
        raise RuntimeError("Error extracting the tar file") from e


def dir_to_tar(dir_path: str | Path) -> Path:
    """Compresses a directory into a tar.gz file.

    This function takes a directory path, validates it, and compresses the directory into a
    `.tar.gz` archive. The resulting archive is created in the same location as the directory.

    Args:
        dir_path (str | Path): The path to the directory to be compressed.

    Returns:
        Path: The path to the resulting compressed `.tar.gz` file.

    Raises:
        FileNotFoundError: If the specified directory does not exist or is not accessible.
        NotADirectoryError: If the specified path is not a directory.
        RuntimeError: If an unexpected error occurs during compression.

    """
    dir_path = to_path_object(dir_path)

    # Ensure the path exists and is a directory
    if not dir_path.is_dir():
        raise NotADirectoryError(f"The specified path is not a directory: {dir_path}")

    tar_filepath = dir_path.with_suffix(".tar.gz")

    try:
        with tarfile.open(tar_filepath, "w:gz") as tar:
            tar.add(dir_path, arcname="")
    except tarfile.TarError as err:
        raise RuntimeError(f"Failed to compress the directory {dir_path} due to a tarfile error: {err}") from err
    except Exception as err:
        raise RuntimeError(f"An unexpected error occurred while compressing directory {dir_path}: {err}") from err

    return tar_filepath


def http_get(url: str, path: str) -> None:
    """Download a URL to a given path on disk.

    Args:
        url (str): The URL to download.
        path (str): The path to save the downloaded file.

    Raises:
        requests.HTTPError: If the HTTP request returns a non-200 status code.

    Returns:
        None

    """
    if os.path.dirname(path) != "":
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        logging.info(f"Exception when trying to download {url}. Response {req.status_code}")
        req.raise_for_status()
        return

    download_filepath = path + "_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)

    os.rename(download_filepath, path)
    progress.close()


def remove_object(filepath: str | Path | list[str] | list[Path]) -> None:
    """Safely removes a file or a list of files from the filesystem.

    Args:
        filepath (str | Path | list): The path to the file or a list of file paths to be removed.
                                    It can be a string, a pathlib.Path object, or a list of these.

    Raises:
        ValueError: If the filepath is not a string, pathlib.Path, or list.
        FileNotFoundError: If the file does not exist.
        Exception: If an error occurs during file removal.

    """

    def _remove_file(fp: str | Path) -> None:
        if not isinstance(fp, (Path, str)):
            raise ValueError("filepath must be a pathlib.Path or a string.")

        if not os.path.exists(fp):
            raise FileNotFoundError(f"File not found: {fp}")

        try:
            if isinstance(fp, Path):
                fp.unlink()
            else:
                os.remove(fp)
        except Exception as e:
            raise e

    if isinstance(filepath, list):
        for path in filepath:
            _remove_file(path)
    else:
        _remove_file(filepath)


def filetype_filter(extensions: tuple[str, ...]) -> Callable[[str | Path], str | Path | None]:
    """Return a function that filters files their extension.

    Args:
        extensions (tuple[str, ...]): A tuple of file extensions to filter by.

    Returns:
        filter_extension: Callable[[str | Path], T]: A function that takes a directory
            and returns a list of files with the specified extensions.

    """

    def filter_extension(filename: str | Path) -> str | Path | None:
        """Filter files by their extension."""
        return filename if Path(filename).as_posix().endswith(extensions) else None

    return filter_extension


def glob_filetypes(source: str | Path, patterns: Iterable[str], recursive: bool = True) -> list[str]:
    """Include files from the given directory or directories that match the specified file types.

    Args:
        source (str | list[str]): A single directory path or a list of directory paths to search for files.
        patterns (Iterable[str]): A list of file extensions to include (e.g., [".txt", ".csv"]).
        recursive (bool): If True, search for files recursively in subdirectories under source.

    Returns:
        list[str]: A list of Path objects representing the files that match the specified file types.

    Examples:
        >>> glob_filetypes("my_dir", (".txt", ".py"))
        ['my_dir/my_file.txt', 'my_dir/my_script.py']

    """
    suffix = "**/*" if recursive else "*"
    inc_files = [glob(f"{source}/{suffix}{ext}", recursive=recursive) for ext in patterns]
    return list(chain.from_iterable(inc_files))


def condition_filepath_on_pattern(fp: str, /, pattern_start: list[str], pattern_end: list[str]) -> bool:
    """Check if a given file path falls between any of the specified start and end patterns.

    Args:
        fp (str): The file path to check.
        pattern_start (list[str]): A list of start patterns.
        pattern_end (list[str]): A list of end patterns.

    Returns:
        bool: True if the file path falls between any of the start and end patterns, False otherwise.

    Raises:
        ValueError: If the lengths of pattern_start and pattern_end do not match.

    """
    pattern_start = pattern_start if isinstance(pattern_start, list) else list(pattern_start)
    pattern_end = pattern_end if isinstance(pattern_start, list) else list(pattern_end)

    if len(pattern_start) != len(pattern_end):
        raise ValueError("Pattern list start and end must have the same length")

    return any((p1 < fp < p2) for p1, p2 in zip(pattern_start, pattern_end))
