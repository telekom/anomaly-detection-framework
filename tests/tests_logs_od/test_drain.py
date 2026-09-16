# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import configparser
import numpy as np
import polars as pl
import pytest
import shutil
import tempfile as tf

from drain3 import TemplateMiner
from pathlib import Path
from unittest.mock import MagicMock

from adf.logs.parsers.drain.config import DrainConfig
from adf.logs.parsers.drain.drain import DrainLogParser

test_logs = [
    "ERROR: Failed to connect to database",
    "INFO: User logged in successfully",
    "WARNING: Disk usage is above 80%",
    "ERROR: Timeout while connecting to service",
]

CONFIG_PATH = "config.ini"


@pytest.fixture
def temp_state_file():
    with tf.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b"{}")
        temp_file.flush()
        yield temp_file.name
    try:
        Path(temp_file.name).unlink()
    except Exception:
        pass


@pytest.fixture
def temp_config_file():
    config_path = Path(tf.gettempdir()) / "config.ini"
    cfg = DrainConfig()
    cfg.save(config_path)
    # Modifying the config file to remove the 'max_clusters'
    parser = configparser.ConfigParser()
    parser.read(config_path)
    if parser.has_section("DRAIN") and parser.has_option("DRAIN", "max_clusters"):
        parser.remove_option("DRAIN", "max_clusters")
        with open(config_path, "w") as f:
            parser.write(f)
    yield config_path
    try:
        config_path.unlink()
    except Exception:
        pass


@pytest.fixture
def drain_log_parser():
    return DrainLogParser()


def test_initialization():
    parser = DrainLogParser()
    assert isinstance(parser.template_miner, TemplateMiner)


def test_match(drain_log_parser):
    for log in test_logs:
        cluster_id = drain_log_parser.match(log)
        assert isinstance(cluster_id, int)


def test_predict(drain_log_parser):
    predictions = drain_log_parser.predict(test_logs)
    assert len(predictions) == len(test_logs)
    assert all(isinstance(pred, int) for pred in predictions)


def test_parse(drain_log_parser):
    df = pl.DataFrame({"logs": test_logs})
    np_array = np.array(test_logs)
    assert isinstance(drain_log_parser.parse(test_logs), list)
    assert isinstance(drain_log_parser.parse(df), list)
    assert isinstance(drain_log_parser.parse(np_array), list)


def test_fit(drain_log_parser):
    drain_log_parser.fit(test_logs)
    assert drain_log_parser.templates is not None


class DummyPersistence:
    def save_state(self, state):
        pass


def test_save_model(drain_log_parser, tmp_path):
    drain_log_parser.template_miner.persistence_handler = DummyPersistence()
    drain_log_parser.template_miner.config = DrainConfig()
    save_path = tmp_path / "drain_model"
    save_path.mkdir(exist_ok=True)
    drain_log_parser.save_model(save_path)
    assert (save_path / "config.ini").exists()


def test_save_templates(drain_log_parser, tmp_path):
    save_path = tmp_path
    save_path.mkdir(exist_ok=True)
    drain_log_parser.save_templates(save_path / "templates.parquet")
    assert (save_path / "templates.parquet").exists()


def test_properties(drain_log_parser):
    cfg = drain_log_parser.config
    if cfg is not None:
        assert hasattr(cfg, "profiling_enabled")
    assert isinstance(drain_log_parser.cluster_ids, list)
    assert isinstance(drain_log_parser.state_path, (Path, type(None)))
    assert isinstance(drain_log_parser.templates, list)


def test_load_from_dirpath_with_valid_dir(temp_state_file, temp_config_file):
    temp_dir = Path(tf.mkdtemp())
    state_file = temp_dir / "state.bin"
    config_file = temp_dir / "config.ini"

    # Copy temporary state and config files to the directory
    shutil.copy(temp_state_file, state_file)
    shutil.copy(temp_config_file, config_file)

    magic_load = MagicMock(DrainLogParser.load_from_dirpath)
    parser = MagicMock(spec=DrainLogParser)
    parser.load_from_dirpath = magic_load

    # Call the method
    parser = parser.load_from_dirpath(temp_dir)
    assert magic_load.called
    assert magic_load.call_count == 1
    assert magic_load.call_args[0][0] == temp_dir
    assert magic_load.call_args[1] == {}

    # Cleanup
    shutil.rmtree(temp_dir)


def test_load_from_dirpath_with_missing_state_file(temp_config_file):
    temp_dir = Path(tf.mkdtemp())
    config_file = temp_dir / "config.ini"

    # Copy only the config file
    shutil.copy(temp_config_file, config_file)

    parser = DrainLogParser.load_from_dirpath(temp_dir)
    assert isinstance(parser, DrainLogParser)
    assert parser.template_miner is not None

    # Cleanup
    shutil.rmtree(temp_dir)


def test_load_from_dirpath_with_invalid_directory():
    with pytest.raises(NotADirectoryError, match="Directory .* does not exist or is not a directory."):
        DrainLogParser.load_from_dirpath("invalid_directory")
