# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import logging

from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MaxMetric, MeanMetric
from typing import Any
from typing_extensions import override

from adf.core.lightning.metrics import MeanQuantileMetric
from adf.core.lightning.plmodel import PLModel
from adf.core.lightning.utils import construct_loss

from .lstm import LSTMAutoEncoder

logger = logging.getLogger(__name__)


class MetricsAutoEncoder(PLModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        enc_dropout: float | None = None,
        dec_dropout: float | None = None,
        sigma: float = 0.0,
        batch_first: bool = True,
        return_hidden: bool = False,
        loss: str = "MSELoss",
        optimizer: str = "AdamW",
        optimizer_config: dict[str, Any] | None = None,
        scheduler: str | None = None,
        scheduler_config: dict[str, Any] | None = None,
    ):
        """Initialize the AutoEncoder model with the specified parameters.

        Args:
            input_dim (int): The number of input features.
            hidden_dim (int): The number of features in the hidden state.
            num_layers (int): The number of recurrent layers.
            dropout (float, optional): Dropout on metrics features before encoding. Default is 0.0.
            enc_dropout (float | None, optional): Encoder LSTM inter-layer dropout. Falls back to
                ``dropout`` when None. Default is None.
            dec_dropout (float | None, optional): Decoder LSTM inter-layer dropout. Falls back to
                ``dropout`` when None. Default is None.
            sigma (float, optional): Std for multiplicative Gaussian noise on metrics features
                (training only). Default is 0.0.
            batch_first (bool, optional): If True, then input and output tensors are provided as (batch, seq, feature).
                Default is True.
            return_hidden (bool, optional): If True, the hidden states are returned. Default is False.
            loss (str, optional): The loss function to use. Default is "MSELoss".
            optimizer (str, optional): The optimizer to use. Default is "AdamW".
            optimizer_config (dict[str, Any], optional): Additional keyword arguments for the optimizer.
                Default is dict(lr=1e-4, weight_decay=1e-6).
            scheduler (str | None, optional): The scheduler to use. Default is None.
            scheduler_config (dict[str, Any], optional): Additional keyword arguments for the scheduler.
                Default is dict().

        """
        optimizer_config = optimizer_config or {"lr": 1e-4, "weight_decay": 1e-6}
        scheduler_config = scheduler_config or {}

        super().__init__(
            optimizer=optimizer,
            optimizer_config=optimizer_config,
            scheduler=scheduler,
            scheduler_config=scheduler_config,
        )
        # Saves all init arguments under self.hparams.param_name
        self.save_hyperparameters()

        # Initialize the model
        self.ae = LSTMAutoEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            enc_dropout=enc_dropout,
            dec_dropout=dec_dropout,
            sigma=sigma,
            return_hidden=return_hidden,
        )
        self.return_hidden = return_hidden

        # Loss functions
        self.loss = construct_loss(loss)
        self.predict_loss = construct_loss(loss, loss_kwargs=dict(reduction="none"))

        # Metrics
        self.validation_metrics = MetricCollection(
            {"score_max": MaxMetric(), "score_mean": MeanMetric(), "score_quantile": MeanQuantileMetric()},
            prefix="val_",
        )
        self.test_metrics = self.validation_metrics.clone(prefix="test_")

    def set_regularization(
        self,
        dropout: float | None = None,
        enc_dropout: float | None = None,
        dec_dropout: float | None = None,
        sigma: float | None = None,
    ) -> None:
        """Update regularization parameters on a loaded model (e.g. for finetuning).

        Args:
            dropout (float | None): New input dropout probability for metrics features.
            enc_dropout (float | None): New encoder LSTM inter-layer dropout.
            dec_dropout (float | None): New decoder LSTM inter-layer dropout.
            sigma (float | None): New Gaussian noise sigma for metrics features.

        """
        if dropout is not None:
            self.ae.input_dropout.p = dropout
        if sigma is not None:
            self.ae.gaussian_noise.sigma = sigma
        if enc_dropout is not None:
            self.ae.encoder.lstm.dropout = enc_dropout
        if dec_dropout is not None:
            self.ae.decoder.lstm.dropout = dec_dropout

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Perform a forward pass through the autoencoder model.

        Args:
            x (Tensor): The input tensor to the autoencoder.

        Returns:
            Tensor: The output tensor after passing through the autoencoder.

        """
        out = self.ae.forward(x)
        return out[0], out[1]

    @staticmethod
    def score(x: Tensor, y: Tensor) -> Tensor:
        """Compute the mean prediction score for the given inputs.

        Args:
            x (Tensor): The input data.
            y (Tensor): The target data.

        Returns:
            Tensor: The mean MSE score for each sample in the batch, with shape (B,).

        """
        return nn.functional.mse_loss(x, y, reduction="none").mean([1, 2])

    @override
    def training_step(self, batch: Tensor, batch_idx: Tensor | None) -> dict[str, Tensor]:
        """Perform a single training step.

        Args:
            batch (Tensor): The input batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            Tensor: The computed loss for the batch.

        """
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x_hat, x_metrics = self.forward(x)

        loss = self.loss(x_hat, x_metrics)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}

    @override
    def validation_step(self, batch: Tensor, batch_idx: Tensor | None) -> dict[str, Tensor]:
        """Perform a single validation step.

        Args:
            batch (Tensor): The input batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            Tensor: The computed loss for the batch.

        Raises:
            ValueError: If the predicted output size does not match the input size.

        """
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x_hat, x_metrics = self(x)
        loss = self.loss(x_hat, x_metrics)
        score = self.score(x_hat, x_metrics)

        self.validation_metrics.update(score)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_score": score}

    @override
    def test_step(self, batch: Tensor, batch_idx: Tensor | None) -> dict[str, Tensor]:
        """Perform a single test step.

        Args:
            batch (Tensor): The input batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            Tensor: The computed loss for the batch.

        """
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x_hat, x_metrics = self(x)
        loss = self.loss(x_hat, x_metrics)
        score = self.score(x_hat, x_metrics)

        self.test_metrics.update(score)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"test_loss": loss, "test_score": score}

    @override
    def predict_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Perform a prediction step on a given batch of data.

        Args:
            batch (Tensor): The input data batch.
            batch_idx (int): The index of the batch.

        Returns:
            Tensor: The score computed from the predicted output and metrics.

        """
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x_hat, x_metrics = self(x)
        return self.score(x_hat, x_metrics)

    @override
    def on_validation_epoch_end(self) -> None:
        """Call at the end of the validation epoch.

        This method logs the computed validation metrics and then resets the metrics for the next epoch.

        The metrics are logged with the following parameters:
        - `on_step=False`: Metrics are not logged at each step.
        - `on_epoch=True`: Metrics are logged at the end of the epoch.
        - `prog_bar=True`: Metrics are displayed in the progress bar.
        - `logger=True`: Metrics are logged using the logger.

        After logging, the validation metrics are reset to prepare for the next epoch.
        """
        metrics = self.validation_metrics.compute()
        logger.debug("Validation epoch %d metrics: %s", self.current_epoch, metrics)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    @override
    def on_test_epoch_end(self) -> None:
        """Call at the end of the testt epoch.

        This method logs the computed validation metrics and then resets the metrics for the next epoch.

        The metrics are logged with the following parameters:
        - `on_step=False`: Metrics are not logged at each step.
        - `on_epoch=True`: Metrics are logged at the end of the epoch.
        - `prog_bar=True`: Metrics are displayed in the progress bar.
        - `logger=True`: Metrics are logged using the logger.

        After logging, the validation metrics are reset to prepare for the next epoch.
        """
        metrics = self.test_metrics.compute()
        logger.debug("Test epoch %d metrics: %s", self.current_epoch, metrics)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    @override
    def on_train_end(self) -> None:
        """Log a concise INFO-level summary of the final training metrics."""
        final_metrics = {
            k: f"{v:.4f}" if hasattr(v, "__float__") else v for k, v in self.trainer.logged_metrics.items()
        }
        logger.info("Training complete — final metrics: %s", final_metrics)
