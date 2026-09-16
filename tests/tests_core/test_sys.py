# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import os

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from adf.core.common.sys import (
    create_dirs,
    create_files,
    dir_to_tar,
    http_get,
    replace_dir,
    to_path_object,
    uuid_name,
)


def test_uuid_name():
    """Test the uuid_name function"""
    uuid = uuid_name()
    today = date.today()
    assert uuid.startswith(str(today))
    assert len(uuid.split("_")) == 3  # Ensure the UUID part is present


def test_to_path_object_str():
    """Test to_path_object with a string"""
    path = "test_path"
    path_obj = to_path_object(path)
    assert path_obj == Path(path)


def test_to_path_object_path():
    """Test to_path_object with a Path object"""
    path = Path("test_path")
    path_obj = to_path_object(path)
    assert path_obj == path


def test_create_files():
    """Test create_files function"""
    with TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "testfile.txt")
        create_files(file_path)
        assert os.path.isfile(file_path)

        # Test with list of files
        file_paths = [os.path.join(tmpdir, f"testfile_{i}.txt") for i in range(3)]
        create_files(file_paths)
        for path in file_paths:
            assert os.path.isfile(path)


def test_create_dirs():
    """Test create_dirs function"""
    with TemporaryDirectory() as tmpdir:
        dir_path = os.path.join(tmpdir, "testdir")
        create_dirs(dir_path)
        assert os.path.isdir(dir_path)

        # Test creating multiple directories
        dirs = [os.path.join(tmpdir, f"dir_{i}") for i in range(3)]
        create_dirs(*dirs)
        for dir in dirs:
            assert os.path.isdir(dir)


@patch("shutil.copytree")
def test_replace_dir(mock_copytree):
    """Test replace_dir function"""
    with TemporaryDirectory() as tmpdir:
        source_dir = os.path.join(tmpdir, "source")
        destination_dir = os.path.join(tmpdir, "destination")
        os.makedirs(source_dir)

        # Test replacing directory
        replace_dir(source_dir, destination_dir)
        # Ensure the mock is called with Path objects for consistency
        mock_copytree.assert_called_once_with(Path(source_dir), Path(destination_dir))


def test_dir_to_tar():
    """Test dir_to_tar function"""
    with TemporaryDirectory() as tmpdir:
        dir_path = os.path.join(tmpdir, "test_dir")
        os.makedirs(dir_path)
        tar_file = dir_to_tar(dir_path)
        # Ensure the tar_file is a string and ends with .tar.gz
        assert str(tar_file).endswith(".tar.gz")
        assert os.path.isfile(tar_file)


@patch("requests.get")
def test_http_get(mock_get):
    """Test http_get function"""
    with TemporaryDirectory() as tmpdir:
        url = "http://example.com/testfile"
        file_path = os.path.join(tmpdir, "testfile")
        mock_get.return_value.status_code = 200
        mock_get.return_value.iter_content.return_value = [b"dummy data"]

        # Perform the test
        http_get(url, file_path)
        assert os.path.isfile(file_path)
