# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from adf.core.evaluation.evt import GPDTailParams, evaluate_model_evt, fit_gpd, gpd_threshold


def test_fit_gpd_returns_none_when_no_exceedances() -> None:
    scores = np.array([0.0, 0.0, 0.0], dtype=float)
    assert fit_gpd(scores, u=1.0) is None


def test_fit_gpd_counts_and_pu() -> None:
    scores = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    params = fit_gpd(scores, u=1.5)
    assert params is not None
    assert params.n_total == 4
    assert params.n_exceed == 2
    assert params.pu == pytest.approx(0.5)


def test_gpd_threshold_alpha_ge_pu_returns_u() -> None:
    params = GPDTailParams(u=10.0, xi=0.1, beta=2.0, pu=0.1, n_exceed=10, n_total=100)
    assert gpd_threshold(alpha=0.2, params=params) == pytest.approx(10.0)


def test_evaluate_model_evt_raises_on_small_train() -> None:
    scores_train = np.zeros(1000, dtype=float)
    scores_val = np.zeros(1000, dtype=float)
    with pytest.raises(ValueError, match="more than 1000"):
        evaluate_model_evt(scores_train, scores_val, alpha=1e-3)


def test_evaluate_model_evt_end_to_end_sanity() -> None:
    # Large N + fixed RNG to keep SciPy fit stable across environments.
    rng = np.random.default_rng(0)
    scores_train = rng.exponential(scale=1.0, size=20_000)
    scores_val_norm = rng.exponential(scale=1.0, size=20_000)

    alpha = 1e-3
    res = evaluate_model_evt(scores_train, scores_val_norm, alpha=alpha)
    assert res is not None
    assert np.isfinite(res.T_alpha)
    assert 0.0 <= res.far_val <= 1.0

    # We expect approximate calibration; tolerance accounts for tail fitting noise + candidate selection.
    assert abs(res.far_val - alpha) < 2.5e-3
