# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import psutil
import pytest
import time

from collections import namedtuple

from adf.core.common.logger import log_files_size, log_total_system_disk_usage, setup_logging, timelog


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def list_logger():
    """Provides a logger with a ListHandler to capture log messages."""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    handler = ListHandler()
    logger.addHandler(handler)
    yield logger, handler
    logger.removeHandler(handler)


# ---------------------------------------------------
# Test setup_logging
# ---------------------------------------------------


def test_setup_logging(tmp_path):
    log_file = tmp_path / "test.log"
    setup_logging(str(log_file))
    root_logger = logging.getLogger()

    file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
    assert file_handlers, "No FileHandler attached to the root logger"
    assert any(os.path.abspath(h.baseFilename) == str(log_file.resolve()) for h in file_handlers)

    stream_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
    assert stream_handlers, "No StreamHandler attached to the root logger"

    test_message = "loglog"
    root_logger.info(test_message)
    for h in file_handlers:
        h.flush()
    content = log_file.read_text()
    assert test_message in content


# ---------------------------------------------------
# Test log_total_system_disk_usage
# ---------------------------------------------------


def test_log_total_system_disk_usage(monkeypatch, list_logger):
    logger, handler = list_logger

    FakePartition = namedtuple("FakePartition", ["device", "mountpoint", "fstype", "opts"])

    fake_partition = FakePartition(device="/dev/nvme1example", mountpoint="/fake_mount", fstype="ext4", opts="rw")

    monkeypatch.setattr(psutil, "disk_partitions", lambda all: [fake_partition])

    FakeUsage = namedtuple("FakeUsage", ["total", "used", "free"])
    fake_usage = FakeUsage(total=10 * 1024**3, used=2 * 1024**3, free=8 * 1024**3)
    monkeypatch.setattr(psutil, "disk_usage", lambda path: fake_usage)

    where_str = "TestDiskUsage"
    log_total_system_disk_usage(logger, where_str)

    messages = [record.getMessage() for record in handler.records]
    assert any(where_str in m for m in messages), "The log does not contain the expected 'where' identifier."
    assert any("-" * 40 in m for m in messages), "The separator line was not logged."


# ---------------------------------------------------
# Test log_files_size
# ---------------------------------------------------


def test_log_files_size(tmp_path, list_logger):
    logger, handler = list_logger
    where_str = "TestFileSize"
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file1.write_bytes(b"a" * 1024)  # 1 KB
    file2.write_bytes(b"b" * 2048)  # 2 KB

    files = [file1, file2]
    log_files_size(logger, where_str, files, mem_type="MB")

    messages = [record.getMessage() for record in handler.records]
    total_bytes = file1.stat().st_size + file2.stat().st_size
    expected_value = total_bytes / (1024**2)
    expected_fragment = f"Total size of files: {expected_value:.5f} (MB)"
    assert any(expected_fragment in m for m in messages), "The logged file size does not match the expected value."


# ---------------------------------------------------
# Test timelog decorator
# ---------------------------------------------------


def test_timelog_decorator(monkeypatch, list_logger):
    logger, handler = list_logger

    monkeypatch.setattr("adf.core.common.logger.logger", logger)

    @timelog
    def dummy_function(x, y):
        time.sleep(0.1)
        return x + y

    result = dummy_function(3, 4)
    assert result == 7

    messages = [record.getMessage() for record in handler.records]
    assert any("dummy_function" in m for m in messages), "No log message includes the function name."
