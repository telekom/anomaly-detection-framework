# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch

from collections.abc import Mapping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from typing import Any, cast

from adf.core.lightning.hpo import validate_optuna_cfg
from adf.core.lightning.plmodel import PLModel
from adf.logs.models.autoencoder.model import LogAutoEncoder
from adf.metrics.models.autoencoder.datamodule import MetricsDataModule
from adf.metrics.models.autoencoder.model import MetricsAutoEncoder


def validate_metrics_hpo_cfg(cfg: DictConfig) -> None:
    """Validate that the config has required sections for metrics HPO.

    Args:
        cfg: Hydra/OmegaConf config expected to contain ``optuna``, ``param_search``,
            ``inputs`` (normalized data paths), and ``outputs.params``.

    """
    validate_optuna_cfg(cfg)
    if getattr(cfg, "param_search", None) is None:
        raise ValueError("Missing required config: param_search")
    if not hasattr(cfg.param_search, "train_max"):
        raise ValueError("Missing required config: param_search.train_max")
    if not hasattr(cfg.param_search, "target_updates"):
        raise ValueError("Missing required config: param_search.target_updates")
    if not hasattr(cfg, "outputs") or not hasattr(cfg.outputs, "params"):
        raise ValueError("Missing required config: outputs.params")
    if not hasattr(cfg.outputs.params, "prefix"):
        raise ValueError("Missing required config: outputs.params.prefix")


def build_metrics_datamodule(cfg: DictConfig, train_tensor: Tensor, val_tensor: Tensor) -> MetricsDataModule:
    """Construct a ``MetricsDataModule`` for a given pair of train/val tensors.

    Args:
        cfg: Hydra/OmegaConf config containing the ``datamodule`` section.
        train_tensor: Training windows tensor.
        val_tensor: Validation windows tensor.

    """
    datamodule_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.datamodule, resolve=True))
    return MetricsDataModule(dataset=train_tensor, dataset_val=val_tensor, **datamodule_cfg)


def compute_total_steps(train_size: int, max_epochs: int, batch_size: int) -> int:
    """Compute total scheduler steps from dataset size, epochs, and batch size."""
    steps_per_epoch = max(1, math.ceil(train_size / batch_size))
    return steps_per_epoch * max_epochs


def run_metrics_training(
    cfg: DictConfig,
    train_tensor: Tensor,
    val_tensor: Tensor,
    tensorboard: TensorBoardLogger,
    model_class: type[MetricsAutoEncoder | LogAutoEncoder] = MetricsAutoEncoder,
) -> PLModel:
    """Train a metrics autoencoder once with the given config and dataset.

    Args:
        cfg: Hydra/OmegaConf config used to initialize model, trainer, and datamodule.
        train_tensor: Training windows tensor.
        val_tensor: Validation windows tensor.
        tensorboard: Logger instance for training logs.
        model_class: Class of the model to train.
        Defaults to ``MetricsAutoEncoder``.

    Returns:
        The trained model instance.

    """
    datamodule = build_metrics_datamodule(cfg=cfg, train_tensor=train_tensor, val_tensor=val_tensor)
    datamodule.setup(stage="fit")

    batch_size = cfg.datamodule.get("batch_size", 32)
    train_size = len(train_tensor)
    cfg.autoencoder.scheduler_config.total_steps = compute_total_steps(train_size, cfg.trainer.max_epochs, batch_size)

    model = model_class(**cfg.autoencoder)
    model.to(torch.float32)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    model.fit(
        datamodule=datamodule,
        callbacks=[ModelCheckpoint(**cfg.model_checkpoint), lr_monitor],
        loss_logger=tensorboard,
        **cfg.trainer,
    )
    return model


def get_optuna_metric(metrics: Mapping[str, Any], metric_name: str) -> float:
    """Extract a single scalar metric from Lightning callback metrics.

    Args:
        metrics: Mapping of metric names to values (tensors/numbers).
        metric_name: The key to read from ``metrics``.

    Returns:
        The metric value as float.

    """
    metric_value = metrics.get(metric_name)
    if metric_value is None:
        available = ", ".join(sorted(metrics.keys()))
        raise ValueError(f"Optuna metric '{metric_name}' not found. Available: {available}")
    return float(metric_value)
