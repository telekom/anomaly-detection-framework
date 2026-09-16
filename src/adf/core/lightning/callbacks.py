# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import lightning as L
import logging

from torch import Tensor
from torch_ema import ExponentialMovingAverage
from typing import Any
from typing_extensions import override

CHECKPOINT_NAME_LAST = "last"
CKPT_FILE_EXTENSION = ".ckpt"
PTH_FILE_EXTENSION = ".pth"
CHECKPOINT_JOIN_CHAR = "-"
STARTING_VERSION = "v1"


class EMAUpdateCallback(L.Callback):
    def __init__(self, average_coef: float | Tensor):
        """Initialize the callback with the given average coefficient.

        Args:
            average_coef (float): The coefficient used for averaging.

        """
        super().__init__()

        self.average_coef = average_coef
        self.ema = None

    @override
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Call at the end of each training epoch to update the EMA.

        This function updates the Exponential Moving Average (EMA) of the model parameters
        if the average coefficient is greater than 0. If the EMA object is not already
        instantiated, it will be created using the model parameters and the specified
        decay rate. If the EMA object already exists, it will be updated with the current
        model parameters.

        Args:
            trainer: The trainer object that is handling the training process.
            pl_module: The PyTorch Lightning module that is being trained.

        """
        if self.average_coef > 0:
            if self.ema is None:
                logging.debug("Initializing ExponentialMovingAverage object")
                self.ema = ExponentialMovingAverage(pl_module.parameters(), decay=self.average_coef)
            else:
                logging.debug("Updating ExponentialMovingAverage object")
                self.ema.update()
                self.ema.copy_to()

    @override
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Call at the end of the training process to reset the EMA object.

        Args:
            trainer (L.Trainer): The trainer instance that was used for training.
            pl_module (L.LightningModule): The model that was being trained.

        Returns:
            None

        """
        logging.debug("Resetting ExponentialMovingAverage object")
        self.ema = None


class LoggedMetricsSaveCallback(L.Callback):
    def __init__(self, monitor: str | list[str] | None = None):
        """Initialize the callback with the given monitor."""
        self.monitor = monitor

    @override
    def on_save_checkpoint(self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict[str, Any]) -> None:
        """Save the metrics at the checkpoint on the end of the training loop.

        Args:
            trainer (L.Trainer): The trainer object that is handling the training process.
            pl_module (L.LightningModule): The PyTorch Lightning module that is being trained.
            checkpoint (dict): The checkpoint object that is being saved.

        """
        metrics = trainer.logged_metrics
        if isinstance(self.monitor, str):
            checkpoint.update({self.monitor: metrics.get(self.monitor, None)})
        elif isinstance(self.monitor, list):
            for monitor in self.monitor:
                checkpoint.update({monitor: metrics.get(monitor, None)})
        else:
            raise TypeError(f"Invalid monitor type: {self.monitor}. Supported monitors are str or list.")
