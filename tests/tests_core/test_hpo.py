# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from omegaconf import DictConfig, OmegaConf

from adf.core.lightning.hpo import run_optuna_multiprocess, run_optuna_study

optuna = pytest.importorskip("optuna")


def _base_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "optuna": {
                "n_trials": 2,
                "n_jobs": 1,
                "n_workers": 1,
                "direction": "minimize",
                "study_name": "test_hpo_core",
                "storage": None,
            }
        }
    )


def test_run_optuna_study_smoke() -> None:
    cfg = _base_cfg()

    def objective(trial: optuna.Trial, _: DictConfig) -> float:
        return float(trial.number)

    study = run_optuna_study(cfg, objective=objective)
    assert study.best_trial is not None


def test_multiprocess_requires_storage() -> None:
    cfg = _base_cfg()
    cfg.optuna.n_workers = 2
    cfg.optuna.storage = None

    def worker_fn(_: dict[str, object], __: int) -> None:
        raise AssertionError("worker_fn should not run without storage")

    with pytest.raises(ValueError, match="optuna.storage must be set"):
        run_optuna_multiprocess(cfg, worker_fn)
