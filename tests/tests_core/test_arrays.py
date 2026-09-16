# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from adf.core.common.arrays import sliding_window

# ---------------------------------------------------
# Tests for sliding_window
# ---------------------------------------------------


def test_sliding_window_basic():
    arr = np.array([1, 2, 3, 4, 5])
    result = sliding_window(arr, 3)
    expected = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    np.testing.assert_array_equal(result, expected)


def test_sliding_window_window_equals_length():
    arr = np.array([1, 2, 3, 4, 5])
    result = sliding_window(arr, 5)
    expected = np.array([[1, 2, 3, 4, 5]])
    np.testing.assert_array_equal(result, expected)
