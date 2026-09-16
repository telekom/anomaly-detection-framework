# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

"""Shared type aliases for input drift monitoring."""

from __future__ import annotations

from sklearn.preprocessing import RobustScaler
from typing import Any, TypedDict

ThresholdsDict = dict[str, Any]
Tier0CheckDict = dict[str, Any]
DriftReportDict = dict[str, Any]
FrozenScalerDict = dict[str, RobustScaler | None]


class PsiResult(TypedDict):
    psi: float
    signal: str
    n_bins: int
