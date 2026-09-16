# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from numpy.typing import NDArray
from scipy.stats import genpareto, ks_2samp, kstest, wasserstein_distance
from typing import Optional

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GPDTailParams:
    r"""GPD tail parameters fitted with Peaks-Over-Threshold (POT).

    Attributes:
        u: POT threshold used to define the tail (fit is on scores > u).
        xi: GPD shape parameter (tail index).
        beta: GPD scale parameter (> 0).
        pu: Empirical tail fraction \(P(X > u)\).
        n_exceed: Number of exceedances (scores > u).
        n_total: Total number of samples in the input.

    """

    u: float
    xi: float
    beta: float
    pu: float
    n_exceed: int
    n_total: int


def fit_gpd(scores: FloatArray, u: float) -> GPDTailParams | None:
    """Fit a GPD on exceedances over threshold ``u``.

    The GPD is fitted on the exceedances ``y = scores[scores > u] - u`` with loc fixed to 0.
    Returns ``None`` if there are no exceedances.
    """
    scores = np.asarray(scores, dtype=np.float64)
    exceed = scores[scores > u]
    n_exceed = int(exceed.size)
    n_total = int(scores.size)
    if n_exceed == 0:
        return None

    y = exceed - u  # y > 0

    # SciPy: genpareto.fit(data, floc=0) -> (shape=xi, loc, scale=beta)
    xi, _loc, beta = genpareto.fit(y, floc=0.0)

    pu = n_exceed / n_total
    return GPDTailParams(u=float(u), xi=float(xi), beta=float(beta), pu=float(pu), n_exceed=n_exceed, n_total=n_total)


def gpd_threshold(alpha: float, params: GPDTailParams) -> float:
    """Compute ``T(alpha)`` such that ``P(X > T) = alpha`` using the fitted GPD tail."""
    u, xi, beta, pu = params.u, params.xi, params.beta, params.pu

    # If alpha is larger than the empirical tail fraction, the threshold lies in the body;
    # return u (a conservative, monotone fallback).
    if alpha >= pu:
        return float(u)

    # Tail probability conditional on X > u
    tail_prob = alpha / pu  # P(X > T | X > u)

    if abs(xi) < 1e-6:
        # Exponential tail (xi ≈ 0): tail_prob = exp(-(T - u)/beta)
        t = u - beta * np.log(tail_prob)
    else:
        # General GPD:
        # tail_prob = (1 + xi * (T - u)/beta)^(-1/xi)
        # => T = u + (beta/xi) * (tail_prob^(-xi) - 1)
        t = u + (beta / xi) * (tail_prob ** (-xi) - 1.0)

    return float(t)


@dataclass(frozen=True)
class UCandidateScore:
    """Quality metrics for a candidate POT threshold ``u``."""

    params: GPDTailParams
    T_alpha: float
    far_val: float
    far_error: float
    xi_jump: float
    beta_rel_jump: float
    ks_stat: float


def gpd_ks_stat(y: FloatArray, xi: float, beta: float) -> float:
    r"""KS statistic between exceedances ``y`` and fitted GPD(xi, beta) (loc=0).

    Returns an \(\sqrt{n}\)-scaled KS statistic so values are comparable across sample sizes.
    """
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = int(y.size)
    if n < 20:
        # too few points to trust the test
        return float("inf")

    stat, _pvalue = kstest(y, "genpareto", args=(xi, 0.0, beta))
    return float(stat * np.sqrt(n))


def sweep_u_candidates(
    scores_train: FloatArray,
    scores_val_norm: FloatArray,
    *,
    alpha: float,
    q_min: float = 0.9999,
    q_max: float = 0.99999,
    n_q: int = 50,
    min_exceed: int = 100,
    xi_tol: float = 0.1,
    beta_rel_tol: float = 0.2,
    xi_max: float = 0.7,
) -> list[UCandidateScore]:
    """Sweep candidate POT thresholds ``u`` on high quantiles and keep stable fits.

    For each candidate ``u``:
    - fit a GPD on exceedances,
    - compute the EVT threshold ``T(alpha)``,
    - estimate FAR on validation normals: ``mean(scores_val_norm > T(alpha))``,
    - reject unstable parameter jumps compared to the previously accepted candidate,
    - score tail fit using a one-sample KS statistic.
    """
    scores_train = np.asarray(scores_train, dtype=np.float64)
    scores_val_norm = np.asarray(scores_val_norm, dtype=np.float64)

    quantiles = np.linspace(q_min, q_max, n_q)
    u_values = np.quantile(scores_train, quantiles)

    results: list[UCandidateScore] = []
    prev_params: GPDTailParams | None = None

    for u in u_values:
        params = fit_gpd(scores_train, float(u))
        if params is None or params.n_exceed < min_exceed:
            continue
        if params.xi >= xi_max:
            continue

        exceed = scores_train[scores_train > params.u]
        y = exceed - params.u
        ks_stat = gpd_ks_stat(y, params.xi, params.beta)

        T_alpha = gpd_threshold(alpha, params)

        far_val = float(np.mean(scores_val_norm > T_alpha))
        far_error = abs(far_val - alpha)

        if prev_params is None:
            xi_jump = 0.0
            beta_rel_jump = 0.0
            stable = True
        else:
            xi_jump = abs(params.xi - prev_params.xi)
            beta_rel_jump = abs(params.beta - prev_params.beta) / prev_params.beta if prev_params.beta > 0 else np.inf
            stable = (xi_jump <= xi_tol) and (beta_rel_jump <= beta_rel_tol)

        if not stable:
            continue

        results.append(
            UCandidateScore(
                params=params,
                T_alpha=T_alpha,
                far_val=far_val,
                far_error=far_error,
                xi_jump=xi_jump,
                beta_rel_jump=float(beta_rel_jump),
                ks_stat=ks_stat,
            )
        )
        prev_params = params

    return results


def select_best_u(candidates: list[UCandidateScore], *, near_best_ks_eps: float = 0.2) -> UCandidateScore | None:
    """Select the best candidate by (KS fit + FAR calibration), preferring lower ``u``."""
    if not candidates:
        return None

    best_ks = min(c.ks_stat for c in candidates)
    near_best = [c for c in candidates if c.ks_stat <= best_ks + near_best_ks_eps]
    return sorted(near_best, key=lambda c: (c.far_error, c.params.u))[0]


@dataclass(frozen=True)
class SeparationMetrics:
    ks_stat: float
    ks_pvalue: float
    wasserstein: float


def compute_separation(scores_val_norm: FloatArray, scores_val_anom: FloatArray) -> SeparationMetrics:
    """Compare normal vs anomalous score distributions (KS + Wasserstein)."""
    scores_val_norm = np.asarray(scores_val_norm, dtype=np.float64)
    scores_val_anom = np.asarray(scores_val_anom, dtype=np.float64)

    ks_stat, ks_pvalue = ks_2samp(scores_val_norm, scores_val_anom)
    wdist = wasserstein_distance(scores_val_norm, scores_val_anom)
    return SeparationMetrics(ks_stat=float(ks_stat), ks_pvalue=float(ks_pvalue), wasserstein=float(wdist))


def anomaly_recall(scores_val_anom: FloatArray, T_alpha: float) -> float:
    """Fraction of labeled anomalies above threshold ``T_alpha``."""
    scores_val_anom = np.asarray(scores_val_anom, dtype=np.float64)
    if scores_val_anom.size == 0:
        return float("nan")
    return float(np.mean(scores_val_anom > T_alpha))


@dataclass(frozen=True)
class EVTModelEval:
    params: GPDTailParams
    T_alpha: float
    far_val: float
    far_error: float
    separation: Optional[SeparationMetrics]
    anomaly_recall: Optional[float]


def evaluate_model_evt(
    scores_train: FloatArray, scores_val_norm: FloatArray, *, alpha: float, scores_val_anom: FloatArray | None = None
) -> EVTModelEval | None:
    """Calibrate an anomaly threshold using POT+EVT.

    - Auto-select a stable POT threshold ``u`` via a high-quantile sweep.
    - Fit a GPD to the tail and compute ``T(alpha)``.
    - Estimate FAR on validation normals.
    - Optionally compute separation + recall when anomalous validation scores are provided.
    """
    scores_train = np.asarray(scores_train, dtype=np.float64)
    scores_val_norm = np.asarray(scores_val_norm, dtype=np.float64)

    if scores_train.size < 1001:
        raise ValueError(f"Training scores must have more than 1000 points, got {scores_train.size}")

    q_min = 1 - 1000 / scores_train.size
    q_max = 1 - 100 / scores_train.size

    candidates = sweep_u_candidates(scores_train, scores_val_norm, alpha=alpha, q_min=q_min, q_max=q_max)
    best = select_best_u(candidates)
    if best is None:
        return None

    separation = None
    recall = None
    if scores_val_anom is not None:
        separation = compute_separation(scores_val_norm, scores_val_anom)
        recall = anomaly_recall(scores_val_anom, best.T_alpha)

    return EVTModelEval(
        params=best.params,
        T_alpha=best.T_alpha,
        far_val=best.far_val,
        far_error=best.far_error,
        separation=separation,
        anomaly_recall=recall,
    )
