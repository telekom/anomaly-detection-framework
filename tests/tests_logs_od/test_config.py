# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import configparser
import os
import pytest

from adf.logs.parsers.drain.config import DrainConfig


@pytest.fixture
def default_config():
    return DrainConfig()


def test_default_values(default_config):
    assert default_config.profiling_enabled is False
    assert default_config.profiling_report_sec == 60
    assert default_config.snapshot_interval_minutes == 5
    assert default_config.snapshot_compress_state is True
    assert default_config.drain_extra_delimiters == []
    assert default_config.drain_sim_th == 0.4
    assert default_config.drain_depth == 4
    assert default_config.drain_max_children == 100
    assert default_config.drain_max_clusters is None
    assert default_config.parametrize_numeric_tokens is True


def test_custom_values():
    config = DrainConfig(profiling_enabled=True, drain_depth=10, drain_sim_th=0.7)
    assert config.profiling_enabled is True
    assert config.drain_depth == 10
    assert config.drain_sim_th == 0.7


def test_save_config(tmp_path):
    config = DrainConfig(profiling_enabled=True, drain_sim_th=0.8)
    config_path = tmp_path / "test_config.ini"

    config.save(config_path)

    assert config_path.exists()

    parser = configparser.ConfigParser()
    parser.read(config_path)

    assert parser.getboolean("PROFILING", "enabled") is True
    assert parser.getint("PROFILING", "report_sec") == 60
    assert parser.getfloat("DRAIN", "sim_th") == 0.8


def test_save_config_with_custom_path():
    config = DrainConfig(profiling_enabled=False)
    config_path = "test_config.ini"

    config.save(config_path)

    parser = configparser.ConfigParser()
    parser.read(config_path)

    assert parser.getboolean("PROFILING", "enabled") is False

    os.remove(config_path)
