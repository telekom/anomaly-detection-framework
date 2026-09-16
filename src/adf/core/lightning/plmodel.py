# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import torch

from collections.abc import Mapping
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities.model_helpers import _restricted_classmethod_impl
from lightning.pytorch.utilities.types import _PREDICT_OUTPUT, LRSchedulerConfig, OptimizerLRScheduler
from omegaconf import OmegaConf
from pathlib import Path
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import IO, TYPE_CHECKING, Any, TypeVar
from typing_extensions import Self, override

from adf.core.common.logger import timelog as tl
from adf.core.lightning.utils import clear_stale_mpi_env, construct_optimizer, construct_scheduler

logger = logging.getLogger(__name__)

_T_co = TypeVar("_T_co", covariant=True)

if TYPE_CHECKING:
    _restricted_classmethod = classmethod
else:
    _restricted_classmethod = _restricted_classmethod_impl


class PLModel(LightningModule):
    """Pytorch Lightning model class."""

    _scheduler: LRScheduler | ReduceLROnPlateau | LRSchedulerConfig
    _optimizer: Optimizer

    def __init__(
        self,
        optimizer: str = "AdamW",
        optimizer_config: Mapping[str, Any] | None = None,
        scheduler: str | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
    ):
        """Initialize the model with the specified loss, optimizer, and scheduler configurations.

        Args:
            optimizer (str, optional): The optimizer to use for training. Defaults to "AdamW".
            optimizer_config (Mapping[str, Any], optional): Additional keyword arguments to pass to the optimizer.
                Defaults to an empty dictionary.
            scheduler (str, optional): The scheduler to use for training. Defaults to None.
            scheduler_config (Mapping[str, Any], optional): Additional keyword arguments to pass to the scheduler.
                Defaults to an empty dictionary.

        Note:
            scheduler_config accepts one additional keyword parameter:
            - lr_scheduler_config (dict): Additional parameters related to the learning rate scheduler.
            See: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html

        """
        super().__init__()
        self.optimizer = optimizer
        self.optimizer_config = optimizer_config or dict()
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config or dict()

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure the optimizers and schedulers for the model.

        This method constructs the optimizer using the model's parameters and the specified optimizer configuration.
        If a scheduler is provided, it also constructs the scheduler with the given configuration.

        Returns:
            optimizer: (list): A list containing the optimizer.
            optimizer, scheduler: (tuple[list, list] (optional)):
                A 2nd list containing the scheduler if it is specified.

        """
        self._optimizer = construct_optimizer(self.parameters(), self.optimizer, self.optimizer_config)

        if self.scheduler is not None:
            self._scheduler = construct_scheduler(self._optimizer, self.scheduler, self.scheduler_config)
            # # If scheduler is returned as LRSchedulerConfig dict
            # if isinstance(self._scheduler, dict):
            #     # Add optimizer reference that Lightning expects
            #     self._scheduler['optimizer'] = self._optimizer
            #     return {'optimizer': self._optimizer, 'lr_scheduler': self._scheduler}
            # else:
            return [self._optimizer], [self._scheduler]
        else:
            return [self._optimizer]

    @tl
    def fit(
        self,
        datamodule: LightningDataModule | None = None,
        train_dataloaders: list[DataLoader[_T_co]] | None = None,
        val_dataloaders: list[DataLoader[_T_co]] | None = None,
        accelerator: str | Accelerator = "auto",
        loss_logger: TensorBoardLogger | None = None,
        max_epochs: int = 100,
        default_root_dir: str | Path | None = None,
        precision: _PRECISION_INPUT = "32-true",
        accumulate_grad_batches: int = 1,
        check_val_every_n_epoch: int = 1,
        num_sanity_val_steps: int = 0,
        deterministic: bool = False,
        enable_checkpointing: bool = False,
        callbacks: list[Callback] | None = None,
    ) -> None:
        """Fit the model using the provided data module and training configurations.

        Args:
            datamodule (L.LightningDataModule): The data module containing training and validation data.
            train_dataloaders (list[DataLoader[_T_co]], optional): List of training data loaders. Defaults to None.
            val_dataloaders (list[DataLoader[_T_co]], optional): List of validation data loaders. Defaults to None.
            accelerator (Optional[str], optional): The type of accelerator to use (e.g., 'cpu', 'gpu').
                Defaults to None.
            loss_logger (TensorBoardLogger, optional): Logger instance for logging training metrics. Defaults to None.
            max_epochs (int, optional): Maximum number of epochs for training. Defaults to 100.
            default_root_dir (str, optional): Directory to save logs and checkpoints. Defaults to None.
            precision (str, optional): Precision to use for training (e.g., 'bfloat16'). Defaults to "bfloat16".
            num_sanity_val_steps (int, optional): Number of sanity validation steps to run before training.
                Defaults to 0.
            enable_checkpointing (bool, optional): Whether to enable checkpointing during training. Defaults to False.
            callbacks (Optional[list[L.Callback]], optional): list of callbacks to use during training.
                Defaults to None.
            deterministic (bool, optional): Whether to set the random seed for deterministic training.
                Defaults to False.
            accumulate_grad_batches (int, optional): Number of batches to accumulate gradients over. Defaults to 1.
            check_val_every_n_epoch (int, optional): Number of epochs to check validation. Defaults to 1.

        Returns:
            None

        """
        if any([train_dataloaders, val_dataloaders]) and datamodule is not None:
            raise ValueError("Cannot provide both datamodule and dataloaders. Please provide only one.")

        logger.info(
            "Starting training: max_epochs=%d, accelerator=%s, precision=%s", max_epochs, accelerator, precision
        )
        clear_stale_mpi_env()

        self._trainer = Trainer(
            default_root_dir=default_root_dir,
            accelerator=accelerator,
            max_epochs=max_epochs,
            logger=loss_logger,
            precision=precision,
            num_sanity_val_steps=num_sanity_val_steps,
            enable_checkpointing=enable_checkpointing,
            callbacks=callbacks,
            accumulate_grad_batches=accumulate_grad_batches,
            check_val_every_n_epoch=check_val_every_n_epoch,
            deterministic=deterministic,
        )

        self._trainer.fit(
            model=self, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders, datamodule=datamodule
        )

        logger.info("Training finished: completed_epochs=%d", self._trainer.current_epoch)

    @tl
    def test(self, datamodule: LightningDataModule, ckpt_path: str | None = None) -> list[Mapping[str, float]]:
        """Test the model using the provided data module and model checkpoint.

        Args:
            datamodule (L.LightningDataModule): The data module containing testing data.
            ckpt_path (str): Path to the model checkpoint to use for testing.

        """
        return self.trainer.test(self, datamodule=datamodule, ckpt_path=ckpt_path)

    @tl
    def validate(self, datamodule: LightningDataModule, ckpt_path: str | None = None) -> list[Mapping[str, float]]:
        """Validate the model using the provided data module and model checkpoint.

        Args:
            datamodule (L.LightningDataModule): The data module containing testing data.
            ckpt_path (str): Path to the model checkpoint to use for testing.

        """
        return self.trainer.validate(self, datamodule=datamodule, ckpt_path=ckpt_path)

    @tl
    def predict(
        self,
        dataloaders: Any | LightningDataModule | None = None,
        datamodule: LightningDataModule | None = None,
        ckpt_path: str | None = None,
    ) -> _PREDICT_OUTPUT | None:
        """Predict using the provided data module and model checkpoint.

        Args:
            dataloaders (Any | LightningDataModule): The dataloader containing data to predict.
            datamodule (L.LightningDataModule): The data module containing testing data.
            ckpt_path (str): Path to the model checkpoint to use for testing.
                Defaults to True.

        """
        preds = self.trainer.predict(self, dataloaders=dataloaders, datamodule=datamodule, ckpt_path=ckpt_path)
        return preds

    @override
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """In case of a compiled model checkpoint, save the keys without including _orig_mod in them.

        Args:
            checkpoint: Dictionary containing the model checkpoint.

        """
        sd = checkpoint["state_dict"]
        sd = {k.replace("._orig_mod", ""): v for k, v in sd.items()}
        checkpoint["state_dict"] = sd

    @override
    @_restricted_classmethod
    def load_from_checkpoint(
        cls,  # noqa
        checkpoint_path: _PATH | IO[str],
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: _PATH | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Self:
        """Load a model instance from a .pth file and a YAML file containing hyperparameters.

        Args:
            checkpoint_path (str | Path | IO): Path to checkpoint. This can also be a URL, or file-like object
            map_location: If your checkpoint saved a GPU model and you now load on CPUs or a different number of GPUs,
                use this to map to the new setup. The behaviour is the same as in :func:`torch.load`.
            hparams_file: Optional path to a ``.yaml`` or ``.csv`` file containinig the hyperparams with
                hierarchical structure.
            strict (bool): Whether to strictly enforce that the keys in :attr:`checkpoint_path` match the keys
                returned by this module's state dict. Defaults to ``True`` unless ``LightningModule.strict_loading`` is
                set, in which case it defaults to the value of ``LightningModule.strict_loading``.
            **kwargs: Any extra keyword args needed to init the model. Can also be used to override saved
                hyperparameter values.

        Returns:
            LogAutoEncoder: An instance of the LogAutoEncoder class with the loaded model and hyperparameters.

        """
        checkpoint_path = Path(checkpoint_path)  # type: ignore[arg-type]

        if checkpoint_path.joinpath("config.yaml").exists():
            cfg = dict(OmegaConf.load(checkpoint_path.joinpath("config.yaml")))
            _hparams = (
                dict(OmegaConf.load(checkpoint_path.joinpath(hparams_file)))  # type: ignore[arg-type]
                if (hparams_file := cfg.get("hparams_file", "")) != ""
                else {}
            )

            instance = cls(**_hparams)
            model = torch.load(checkpoint_path.joinpath(cfg.get("model", "model.pth")), map_location=map_location)  # type: ignore[arg-type]
            instance.ae = model
            return instance

        return super().load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            **kwargs,
        )
