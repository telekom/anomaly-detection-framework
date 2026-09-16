# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from omegaconf import OmegaConf
from pathlib import Path

from adf.core.common.sys import uuid_name


def omegaconf_enable_resolvers() -> None:
    """Register custom resolvers for OmegaConf, enabling dynamic file count retrieval in configurations.

    This function registers the 'listdir' resolver, which counts files in a directory with a specific suffix.

    The resolver accepts:
      - `path` (str): Directory path to search.
      - `suffix` (str): File suffix to filter by.

    Examples:
        >>> config = OmegaConf.create({
        ...     "num_files": "${listdir:/path/to/dir,.txt}"
        ...     "path_to_files": "${uuid:}/path/to/dir"
        ...})
        >>> print(config.num_files)
        3
        >>> print(config.path_to_files)
        'b8f8a0a7/path/to/dir'

    """

    def omegaconf_listdir_resolver(path: str, suffix: str) -> int:
        """Count files in the specified directory with the given suffix.

        Args:
            path (str): Directory path to search.
            suffix (str): File suffix to filter by.

        Returns:
            int: The number of matching files.

        """
        return len(list(Path(path).glob(f"*{suffix}")))

    def omegaconf_uuid_resolver() -> str:
        """Generate a UUID.

        Returns:
            uuid (str): A UUID string.

        """
        return uuid_name()

    OmegaConf.register_new_resolver("listdir", omegaconf_listdir_resolver)
    OmegaConf.register_new_resolver("uuid", omegaconf_uuid_resolver)
