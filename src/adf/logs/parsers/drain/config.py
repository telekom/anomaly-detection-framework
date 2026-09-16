# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import configparser
import json

from drain3.template_miner_config import TemplateMinerConfig
from pathlib import Path
from typing_extensions import Self


class DrainConfig(TemplateMinerConfig):  # type: ignore
    """Configuration class for the Drain log parser.

    Extends the TemplateMinerConfig and provides additional configuration options specific to the Drain log parser.

    Methods:
        __init__(self): Initializes the configuration with default values.
        save(config_filename: str) -> None:
            Saves the current configuration to a file in INI format.

    Attributes:
        profiling_enabled (bool): Indicates if profiling is enabled.
        profiling_report_sec (int): The interval in seconds for profiling reports.
        snapshot_interval_minutes (int): The interval in minutes for taking snapshots.
        snapshot_compress_state (bool): Indicates if the snapshot state should be compressed.
        drain_extra_delimiters (str): Extra delimiters used by the Drain parser.
        drain_sim_th (float): Similarity threshold for the Drain parser.
        drain_depth (int): Depth of the Drain parsing tree.
        drain_max_children (int): Maximum number of children nodes in the Drain parsing tree.
        drain_max_clusters (int): Maximum number of clusters in the Drain parsing tree.
        parametrize_numeric_tokens (bool): Indicates if numeric tokens should be parameterized.

    """

    def __init__(
        self,
        profiling_enabled: bool = False,
        profiling_report_sec: int = 60,
        snapshot_interval_minutes: int = 5,
        snapshot_compress_state: bool = True,
        drain_extra_delimiters: list[str] | None = None,
        drain_sim_th: float = 0.4,
        drain_depth: int = 4,
        drain_max_children: int = 100,
        drain_max_clusters: int | None = None,
        masking_instructions: list[str] | None = None,
        mask_prefix: str = "<",
        mask_suffix: str = ">",
        parameter_extraction_cache_capacity: int = 3000,
        parametrize_numeric_tokens: bool = True,
    ):
        """Initialize the configuration for the log parser.

        Args:
            profiling_enabled (bool): Flag to enable or disable profiling. Default is False.
            profiling_report_sec (int): Interval in seconds for generating profiling reports. Default is 60.
            snapshot_interval_minutes (int): Interval in minutes for taking snapshots. Default is 5.
            snapshot_compress_state (bool): Flag to enable or disable compression of snapshot state. Default is True.
            drain_extra_delimiters (list[str]): List of extra delimiters for the drain algorithm.
                Default is an empty list.
            drain_sim_th (float): Similarity threshold for the drain algorithm. Default is 0.4.
            drain_depth (int): Depth of the drain parsing tree. Default is 4.
            drain_max_children (int): Maximum number of children nodes in the drain parsing tree. Default is 100.
            drain_max_clusters (int | None): Maximum number of clusters for the drain algorithm. Default is None.
            masking_instructions (list[str]): List of instructions for masking sensitive data. Default is an empty list.
            mask_prefix (str): Prefix for masked data. Default is None.
            mask_suffix (str): Suffix for masked data. Default is None.
            parameter_extraction_cache_capacity (int): Capacity of the cache for parameter extraction. Default is 3000.
            parametrize_numeric_tokens (bool): Flag to enable or disable parameterization of numeric tokens.
                Default is True.

        """
        super().__init__()
        self.profiling_enabled = profiling_enabled
        self.profiling_report_sec = profiling_report_sec
        self.snapshot_interval_minutes = snapshot_interval_minutes
        self.snapshot_compress_state = snapshot_compress_state
        self.drain_extra_delimiters = drain_extra_delimiters or []
        self.drain_sim_th = drain_sim_th
        self.drain_depth = drain_depth
        self.drain_max_children = drain_max_children
        self.drain_max_clusters = drain_max_clusters
        self.masking_instructions = masking_instructions or []
        self.mask_prefix = mask_prefix
        self.mask_suffix = mask_suffix
        self.parameter_extraction_cache_capacity = parameter_extraction_cache_capacity
        self.parametrize_numeric_tokens = parametrize_numeric_tokens

    def save(self, config_filename: str | Path) -> None:
        """Save the current configuration to a file.

        Args:
            config_filename (str): The name of the file where the configuration will be saved.

        Returns:
            None

        """
        parser = configparser.ConfigParser()
        parser.add_section("PROFILING")
        parser.set("PROFILING", "enabled", str(self.profiling_enabled))
        parser.set("PROFILING", "report_sec", str(self.profiling_report_sec))

        parser.add_section("SNAPSHOT")
        parser.set("SNAPSHOT", "snapshot_interval_minutes", str(self.snapshot_interval_minutes))
        parser.set("SNAPSHOT", "compress_state", str(self.snapshot_compress_state))

        parser.add_section("DRAIN")
        if self.drain_extra_delimiters:
            parser.set("DRAIN", "extra_delimiters", str(self.drain_extra_delimiters))

        parser.set("DRAIN", "sim_th", str(self.drain_sim_th))
        parser.set("DRAIN", "depth", str(self.drain_depth))
        parser.set("DRAIN", "max_children", str(self.drain_max_children))
        if self.drain_max_clusters:
            parser.set("DRAIN", "max_clusters", str(self.drain_max_clusters))
        parser.set("DRAIN", "parametrize_numeric_tokens", str(self.parametrize_numeric_tokens))

        parser.add_section("MASKING")
        masking_instructions = [
            {"regex_pattern": mi.pattern.replace("%", "%%"), "mask_with": mi.mask_with}
            for mi in self.masking_instructions
        ]
        if self.masking_instructions:
            parser.set("MASKING", "masking", json.dumps(masking_instructions, indent=0))
        parser.set("MASKING", "mask_prefix", str(self.mask_prefix))
        parser.set("MASKING", "mask_suffix", str(self.mask_suffix))

        with open(config_filename, "w") as configfile:
            parser.write(configfile)

    def load(self, path: str | Path) -> Self:
        """Load the configuration from the specified path.

        Args:
            path (str | Path): The path to the configuration file.

        Returns:
            Self: The instance of the class.

        """
        super().load(config_filename=path)
        return self
