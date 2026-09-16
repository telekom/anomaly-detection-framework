# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from omegaconf import OmegaConf

from adf.logs.models.autoencoder.hpo_utils import get_optuna_metric, validate_autoencoder_hpo_cfg


def test_get_optuna_metric_returns_value() -> None:
    metrics = {"val_loss": 0.5}
    assert get_optuna_metric(metrics, "val_loss") == 0.5


def test_validate_autoencoder_hpo_cfg_smoke() -> None:
    cfg = OmegaConf.create(
        {
            "optuna": {"n_trials": 1, "n_jobs": 1, "n_workers": 1},
            "param_search": {"train_max": 10, "target_updates": 5},
            "inputs": {
                "embeddings": {"key": "k", "destination": "/tmp/e.npy"},
                "windows": {"prefix": "p", "destination": "/tmp/w"},
            },
            "outputs": {"params": {"prefix": "out"}},
        }
    )
    validate_autoencoder_hpo_cfg(cfg)
