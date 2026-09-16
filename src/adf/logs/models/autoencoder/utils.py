# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def default_collate_fastindex(batch: Iterable[Any]) -> Iterable[Any]:
    """Default collate function for fast indexing.

    This function is used to collate a batch of items without stacking tensors. It is particularly useful when
    fast indexing is enabled, allowing for efficient data loading without the overhead of tensor stacking.
    The batch is already created inside the Dataset class, so this function simply returns the batch as is.

    Args:
        batch (list): A list of items to collate.

    """
    return batch
