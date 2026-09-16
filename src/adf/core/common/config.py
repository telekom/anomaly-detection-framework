# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from collections.abc import Mapping
from omegaconf import DictConfig, ListConfig, OmegaConf
from pathlib import Path
from typing import Any

from adf.core.common.resolvers import omegaconf_enable_resolvers


def parse_config(
    config: str | Path | Mapping[str, Any] | DictConfig,
    common: str | Path | Mapping[str, Any] | DictConfig | None = None,
    resolve: bool = True,
    update_with: dict[str, Any] | None = None,
    enable_resolvers: bool = True,
) -> DictConfig | ListConfig:
    """Parse and merge configuration files or dictionaries."""
    if not isinstance(config, (str, Path, Mapping, DictConfig)):
        raise TypeError("config must be a PathLike, Mapping, or DictConfig")

    if common and not isinstance(common, (str, Path, Mapping, DictConfig)):
        raise TypeError("common must be a PathLike, Mapping, or DictConfig")

    if enable_resolvers:
        try:
            omegaconf_enable_resolvers()
        except ValueError:
            logging.warning("Omegaconf resolvers cannot be enabled or already enabled. Passing")

    cfg = OmegaConf.load(config) if isinstance(config, (str, Path)) else config
    if update_with:
        if isinstance(cfg, (DictConfig, ListConfig)):
            cfg = OmegaConf.merge(cfg, OmegaConf.create(update_with))
        else:
            raise TypeError("config must be a DictConfig or ListConfig to update with a dictionary")

    cmn = OmegaConf.load(common) if common and isinstance(common, (str, Path)) else common
    cfg = OmegaConf.merge(cmn, cfg)

    if resolve:
        OmegaConf.resolve(cfg)
    return cfg
