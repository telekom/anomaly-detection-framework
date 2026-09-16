# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch

from collections.abc import Mapping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from typing import Any, cast

from adf.core.lightning.callbacks import LoggedMetricsSaveCallback
from adf.core.lightning.hpo import validate_optuna_cfg
from adf.logs.models.autoencoder.datamodule import LogDataModule
from adf.logs.models.autoencoder.model import LogAutoEncoder

WindowsArray = NDArray[np.int_]
EmbeddingArray = NDArray[np.float_]


def _compute_total_steps(train_size: int, max_epochs: int, batch_size: int) -> int:
    """Compute total scheduler steps (matches pretraining schedule: ~10% warmup buffer)."""
    return int((train_size * max_epochs) // batch_size * 11 // 10)


def validate_autoencoder_hpo_cfg(cfg: DictConfig) -> None:
    """Validate that the config contains the minimum sections for autoencoder HPO.

    Args:
        cfg: Hydra/OmegaConf config expected to contain `optuna`, `param_search`,
            `inputs` (windows/embeddings), and `outputs.params`.

    """
    validate_optuna_cfg(cfg)
    if getattr(cfg, "param_search", None) is None:
        raise ValueError("Missing required config: param_search")
    if not hasattr(cfg.param_search, "train_max"):
        raise ValueError("Missing required config: param_search.train_max")
    if not hasattr(cfg.param_search, "target_updates"):
        raise ValueError("Missing required config: param_search.target_updates")
    if not hasattr(cfg, "inputs") or not hasattr(cfg.inputs, "embeddings"):
        raise ValueError("Missing required config: inputs.embeddings")
    if not hasattr(cfg.inputs.embeddings, "key"):
        raise ValueError("Missing required config: inputs.embeddings.key")
    if not hasattr(cfg.inputs.embeddings, "destination"):
        raise ValueError("Missing required config: inputs.embeddings.destination")
    if not hasattr(cfg.inputs, "windows"):
        raise ValueError("Missing required config: inputs.windows")
    if not hasattr(cfg.inputs.windows, "prefix"):
        raise ValueError("Missing required config: inputs.windows.prefix")
    if not hasattr(cfg.inputs.windows, "destination"):
        raise ValueError("Missing required config: inputs.windows.destination")
    if not hasattr(cfg, "outputs") or not hasattr(cfg.outputs, "params"):
        raise ValueError("Missing required config: outputs.params")
    if not hasattr(cfg.outputs.params, "prefix"):
        raise ValueError("Missing required config: outputs.params.prefix")


def build_datamodule(
    cfg: DictConfig, win_dataset: Mapping[str, WindowsArray], embeddings: EmbeddingArray
) -> LogDataModule:
    """Construct a `LogDataModule` for a given window/embedding dataset.

    Args:
        cfg: Hydra/OmegaConf config containing the `datamodule` section.
        win_dataset: Mapping with at least `train` and `val` window indices.
        embeddings: Embedding matrix aligned with window indices.

    """
    datamodule_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.datamodule, resolve=True))
    # Reason: OmegaConf returns `Any`; cast to mapping for `**` expansion.
    return LogDataModule(windows=win_dataset, embeddings=embeddings, **datamodule_cfg)


def run_training(
    cfg: DictConfig,
    win_dataset: Mapping[str, WindowsArray],
    embeddings: EmbeddingArray,
    train_size: int,
    tensorboard: TensorBoardLogger,
) -> LogAutoEncoder:
    """Train a logs autoencoder once with the given config and dataset.

    Args:
        cfg: Hydra/OmegaConf config used to initialize model, trainer and datamodule.
        win_dataset: Window indices for training/validation.
        embeddings: Embedding matrix used by the model.
        train_size: Number of training samples (used for schedule step estimation).
        tensorboard: Logger instance used for training logs.

    Returns:
        The trained `LogAutoEncoder` instance (Lightning wrapper).

    """
    cfg.autoencoder.scheduler_config.total_steps = _compute_total_steps(
        train_size, cfg.trainer.max_epochs, cfg.datamodule.dataloader_kwargs.batch_size
    )
    datamodule = build_datamodule(cfg=cfg, win_dataset=win_dataset, embeddings=embeddings)

    model = LogAutoEncoder(**cfg.autoencoder)
    model.to(torch.float32)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    lmsc = LoggedMetricsSaveCallback(monitor="train_score_max")

    model.fit(
        datamodule=datamodule,
        callbacks=[ModelCheckpoint(**cfg.model_checkpoint), lr_monitor, lmsc],
        loss_logger=tensorboard,
        **cfg.trainer,
    )
    return model


def get_optuna_metric(metrics: Mapping[str, Any], metric_name: str) -> float:
    """Extract a single scalar metric from Lightning callback metrics.

    Args:
        metrics: Mapping of metric names to values (tensors/nums).
        metric_name: The key to read from `metrics`.

    Returns:
        The metric value as float.

    """
    metric_value = metrics.get(metric_name)
    if metric_value is None:
        available = ", ".join(sorted(metrics.keys()))
        raise ValueError(f"Optuna metric '{metric_name}' not found. Available: {available}")
    return float(metric_value)
