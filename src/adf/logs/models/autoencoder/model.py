# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import numpy as np
import torch

from numpy.typing import NDArray
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MaxMetric, MeanMetric
from typing import TYPE_CHECKING, Any, Literal, cast
from typing_extensions import override

if TYPE_CHECKING:
    from shap import Explanation

from adf.core.lightning.metrics import MeanQuantileMetric
from adf.core.lightning.plmodel import PLModel
from adf.core.lightning.utils import construct_loss, reduce

from .registry import model_loader

logger = logging.getLogger(__name__)


class LogAutoEncoder(PLModel):
    def __init__(
        self,
        model_name: str = "LSTMAutoEncoder",
        model_args: dict[str, Any] | None = None,
        loss: str = "MSELoss",
        compile_model: bool = True,
        compilation_mode: Literal["reduce-overhead", "max-autotune"] = "reduce-overhead",
        metrics: MetricCollection | None = None,
        optimizer: str = "AdamW",
        optimizer_config: dict[str, Any] | None = None,
        scheduler: str | None = None,
        scheduler_config: dict[str, Any] | None = None,
        time_dim: int | None = None,
        use_reg_loss: bool = False,
        beta: float = 0.0,
    ):
        """Initialize the AutoEncoder model with the specified parameters.

        Args:
            model_name (str, optional): The name of the model registered in the MODEL_REGISTRY.
                Defaults to "LSTMAutoEncoder".
            model_args (dict[str, Any], optional): Additional keyword arguments for the model.
            loss (str, optional): The loss function to use. Default is "MSELoss".
            compile_model (bool, optional): If True, the model is compiled. Default is True.
            compilation_mode (str, optional): The compilation mode to use. Default is "reduce-overhead".
            metrics (MetricCollection, optional): The metrics to use. Default is None.
            optimizer (str, optional): The optimizer to use. Default is "AdamW".
            optimizer_config (dict[str, Any], optional): Additional keyword arguments for the optimizer.
                Default is dict(lr=1e-4, weight_decay=1e-6).
            scheduler (str | None, optional): The scheduler to use. Default is None.
            scheduler_config (dict[str, Any], optional): Additional keyword arguments for the scheduler.
                Default is dict().
            time_dim (int | None, optional): The dimension of the time encoding. Default is None.
            use_reg_loss (bool, optional): If True, the regularization loss is added to the loss. Default is False.
            beta (float, optional): The beta value for the regularization loss. Default is 0.0.

        """
        super().__init__(
            optimizer=optimizer,
            optimizer_config=optimizer_config,
            scheduler=scheduler,
            scheduler_config=scheduler_config,
        )
        # Saves all init arguments under self.hparams.param_name
        self.save_hyperparameters()
        model_args = model_args or {}

        # Initialize the model
        self.ae = model_loader(model_name=model_name, model_args=model_args)
        self.return_hidden = model_args.get("return_hidden", False)

        self.compile_model = compile_model
        self.compilation_mode = compilation_mode

        # Loss functions
        self.loss = construct_loss(loss, loss_kwargs={"reduction": "none"})

        # Metrics
        self.train_metrics = (
            metrics
            if metrics is not None
            else MetricCollection(
                {"score_max": MaxMetric(), "score_mean": MeanMetric(), "score_quantile": MeanQuantileMetric()},
                prefix="train_",
            )
        )

        self.validation_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        self.validation_loss_metric = MeanMetric()
        self._last_train_score_mean: float | None = None
        self.time_dim = time_dim
        self.use_reg_loss = use_reg_loss
        self.beta = beta

    @override
    def on_fit_start(self) -> None:
        """Call at the start of the fit method.

        This method compiles the model. We use the `on_fit_start` method to ensure
        that the model is moved to GPU first and then compiled

        Sets:
            self.ae (nn.Module): The compiled model.
        """
        if self.compile_model:
            try:
                self.ae = torch.compile(self.ae, mode=self.compilation_mode)  # type: ignore[assignment]
            except Exception as e:
                logging.error(f"Failed to compile the model: {e}")

    def score(
        self,
        input: Tensor,
        target: Tensor,
        dim: int | list[int] | None = None,
        add_relu: bool = False,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute the mean prediction score for the given inputs.

        Args:
            input (Tensor): The input data.
            target (Tensor): The target data.
            add_relu (bool, optional): If True, the ReLU function is applied to the output. Default is False.
            dim (int | tuple[int, ...], optional): Dimension along which to reduce. Defaults to 0.
            mask (Tensor | None): A mask tensor for ignoring timesteps.

        Returns:
            Tensor: The mean prediction score for each sample in the batch, with shape (B,).

        """
        if self.time_dim:
            input = input[..., : -self.time_dim]
            target = target[..., : -self.time_dim]
        score = self.loss(input, target)
        score = reduce(score, mask=mask, dim=dim)

        if add_relu:
            score = nn.functional.relu(score)

        return score  # type: ignore

    @override
    def forward(
        self, x: Tensor, mask: Tensor | None = None, infer: bool = True
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, tuple[Tensor, Tensor]]:
        """Perform a forward pass through the autoencoder model.

        Args:
            x (Tensor): The input tensor to the autoencoder.
            mask (Tensor | None): A mask tensor for ignoring timesteps.
            infer (bool): If True, the model is in inference mode. Default is False.

        Returns:
            Tensor: The output tensor after passing through the autoencoder.

        """
        if self.time_dim and infer:
            if self.return_hidden:
                x_hat, *_ = self.ae.forward(x, mask=mask)
            else:
                x_hat = self.ae.forward(x, mask=mask)

            if isinstance(x_hat, tuple):
                x_hat = x_hat[0]

            x_hat = x_hat[..., : -self.time_dim]
            x_out = x[..., : -self.time_dim]
            return x_hat, x_out
        else:
            return self.ae.forward(x, mask=mask)  # type: ignore

    @override
    def training_step(self, batch: Tensor, batch_idx: Tensor | None) -> dict[str, Tensor]:
        """Perform a single training step.

        Args:
            batch (Tensor): The input batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            Tensor: The computed loss for the batch.

        Raises:
            ValueError: If the predicted output size does not match the input size.

        """
        batch, mask = (batch, None) if not isinstance(batch, tuple) else batch
        # If the batch is a list or tuple, we need to get the first element e.g. TensorDataset returns a tuple of (x, y)
        batch = batch[0] if isinstance(batch, list) and len(batch) == 1 else batch

        if self.return_hidden:
            x_hat, *_ = self.forward(batch, mask=mask, infer=False)
        else:
            x_hat = self.forward(batch, mask=mask, infer=False)

        kl_loss = None

        if isinstance(x_hat, tuple):
            x_hat, kl_loss = x_hat

        if x_hat.size() != batch.size():
            raise ValueError(f"In {batch_idx} the sizes prediction {x_hat.size()} and target {batch.size()} differs")

        loss = self.score(x_hat, batch, add_relu=False, mask=mask)

        if kl_loss and self.use_reg_loss:
            scale = loss.detach() / (kl_loss.detach().mean() + 1e-8)
            loss += self.beta * scale * kl_loss.mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        score = self.score(x_hat, batch, add_relu=True, mask=mask, dim=[1, 2])
        self.train_metrics.update(score)

        return {"loss": loss, "score": score}

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
        batch, mask = (batch, None) if not isinstance(batch, tuple) else batch
        # If the batch is a list or tuple, we need to get the first element e.g. TensorDataset returns a tuple of (x, y)
        batch = batch[0] if isinstance(batch, list) and len(batch) == 1 else batch

        if self.return_hidden:
            x_hat, *_ = self.forward(batch, mask=mask, infer=False)
        else:
            x_hat = self.forward(batch, mask=mask, infer=False)

        if isinstance(x_hat, tuple):
            x_hat, kl_loss = x_hat

        if x_hat.size() != batch.size():
            raise ValueError(f"In {batch_idx} the sizes prediction {x_hat.size()} and target {batch.size()} differs")

        loss = self.score(x_hat, batch, add_relu=False, mask=mask)
        self.validation_loss_metric.update(loss.detach())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        score = self.score(x_hat, batch, add_relu=True, mask=mask, dim=[1, 2])
        self.validation_metrics.update(score)

        return {"val_loss": loss, "val_score": score}

    @override
    def test_step(self, batch: Tensor, batch_idx: Tensor | None) -> dict[str, Tensor]:
        """Perform a single test step.

        Args:
            batch (Tensor): The input batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            Tensor: The computed loss for the batch.

        Raises:
            ValueError: If the predicted output size does not match the input size.

        """
        batch, mask = (batch, None) if not isinstance(batch, tuple) else batch
        # If the batch is a list or tuple, we need to get the first element e.g. TensorDataset returns a tuple of (x, y)
        batch = batch[0] if isinstance(batch, list) and len(batch) == 1 else batch

        if self.return_hidden:
            x_hat, *_ = self.forward(batch, mask=mask, infer=False)
        else:
            x_hat = self.forward(batch, mask=mask, infer=False)

        if isinstance(x_hat, tuple):
            x_hat, kl_loss = x_hat

        if x_hat.size() != batch.size():
            raise ValueError(f"In {batch_idx} the sizes prediction {x_hat.size()} and target {batch.size()} differs")

        test_loss = self.score(x_hat, batch, add_relu=False, mask=mask)

        score = self.score(x_hat, batch, add_relu=True, mask=mask, dim=[1, 2])
        self.test_metrics.update(score)

        return {"test_loss": test_loss, "test_score": score}

    @override
    def predict_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Perform a single prediction step.

        Args:
            batch (Tensor): The input batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            Tensor: The computed loss for the batch.

        """
        batch, mask = (batch, None) if not isinstance(batch, tuple) else batch
        # If the batch is a list or tuple, we need to get the first element e.g. TensorDataset returns a tuple of (x, y)
        batch = batch[0] if isinstance(batch, list) and len(batch) == 1 else batch

        if self.return_hidden:
            x_hat, *_ = self.forward(batch, mask=mask, infer=False)
        else:
            x_hat = self.forward(batch, mask=mask, infer=False)

        if isinstance(x_hat, tuple):
            x_hat, kl_loss = x_hat

        if x_hat.size() != batch.size():
            raise ValueError(f"In {batch_idx} the sizes prediction {x_hat.size()} and target {batch.size()} differs")

        score = self.score(x_hat, batch, add_relu=False, mask=mask, dim=[1, 2])
        return score

    @override
    def on_train_epoch_end(self) -> None:
        """Call at the end of the training epoch.

        This method logs the computed training metrics and then resets the metrics for the next epoch.

        The metrics are logged with the following parameters:
        - `on_step=False`: Metrics are not logged at each step.
        - `on_epoch=True`: Metrics are logged at the end of the epoch.
        - `prog_bar=True`: Metrics are displayed in the progress bar.
        - `logger=True`: Metrics are logged using the logger.

        After logging, the training metrics are reset to prepare for the next epoch.
        """
        metrics = self.train_metrics.compute()
        self._last_train_score_mean = float(metrics["train_score_mean"].detach().cpu())
        logger.debug("Train epoch %d metrics: %s", self.current_epoch, metrics)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.train_metrics.reset()

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
        val_loss = self.validation_loss_metric.compute()
        val_score_mean = metrics["val_score_mean"]
        if self._last_train_score_mean is None:
            val_error = val_loss
        else:
            # Reason: scale the overfit penalty by val_loss so the ratio stays in the same order.
            ratio = val_score_mean / (self._last_train_score_mean + 1e-8)
            val_error = val_loss * (1.0 + torch.relu(ratio - 1.0))
        self.log("val_error", val_error, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        logger.debug("Validation epoch %d metrics: %s", self.current_epoch, metrics)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.validation_metrics.reset()
        self.validation_loss_metric.reset()

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
        self.test_metrics.reset()

    @override
    def on_train_end(self) -> None:
        """Log a concise INFO-level summary of the final training metrics."""
        final_metrics = {
            k: f"{v:.4f}" if hasattr(v, "__float__") else v for k, v in self.trainer.logged_metrics.items()
        }
        logger.info("Training complete — final metrics: %s", final_metrics)

    @property
    def is_compiled(self) -> bool:
        """Returns True if the model is compiled, otherwise False.

        Returns:
            bool: True if the model is compiled, otherwise False.

        """
        return isinstance(self.ae, torch._dynamo.eval_frame.OptimizedModule)

    @property
    def model_dtype(self) -> torch.dtype | None:
        """Return the model dtype if the model has the dtype attribute.

        Returns:
            torch.dtype: The dtype of the model.

        """
        return getattr(self.ae, "dtype", None)

    def predict_single(
        self, x: Tensor, mask: Tensor | None = None, aggregate: Literal["all", "step", "embedding"] = "all"
    ) -> Tensor:
        """Perform a forward pass through the autoencoder model.

        Args:
            x (Tensor): The input tensor to the autoencoder.
            mask (Tensor): The masking tensor for ignoring timesteps.
            aggregate (str): The aggregation method to use. Can be "step", "embedding", or "all".

        Returns:
            Tensor: The output tensor after passing through the autoencoder.

        """
        if aggregate not in ("step", "embedding", "all"):
            raise ValueError("The aggregate parameter must be either 'step' or 'embedding'.")

        dim = [1] if aggregate == "embedding" else [2] if aggregate == "step" else [1, 2]

        with torch.no_grad():
            if self.return_hidden:
                x_hat, *_ = self.forward(x, infer=False)
            else:
                x_hat = self.forward(x, infer=False)

            if isinstance(x_hat, tuple):
                x_hat, kl_loss = x_hat

            pred = self.score(x_hat, x, mask=mask, dim=dim)
        return pred

    def explain(self, x: Tensor, max_evals: str | int = "auto", **kwargs: Any) -> Explanation:
        """Explain the model predictions using SHAP.

        The default case is to try to explain each step's embedding contribution to the final score.
        The input tensor is masked along the step dimension, and the SHAP values are computed.

        Args:
            x (Tensor): The input tensor to the autoencoder.
            max_evals (str | int): The maximum number of evaluations to perform. Default is "auto".
            **kwargs: Additional keyword arguments for the SHAP explainer.

        Returns:
            (Explanation): The SHAP values for the input tensor.

        """
        import shap

        if x.ndim != 3:
            raise ValueError("The input tensor must be 3-dimensional (batch, steps, features).")

        if x.size(0) != 1:
            raise ValueError("The input tensor must have a batch size of 1 for the explain method.")

        self._x = x.detach().numpy()

        placeholder = torch.zeros(size=(x.shape[0], x.shape[1]))  # (1, steps)
        feature_names = ["step_" + str(i) for i in range(x.shape[1])]

        def _explain_forward(x_in: Tensor | NDArray[np.float_]) -> Tensor:
            """Perform a forward pass through the autoencoder model for SHAP explanation.

            Args:
                x_in (Tensor): The input tensor to the autoencoder.

            Returns:
                Tensor: The output tensor after passing through the autoencoder.

            """
            if isinstance(x_in, np.ndarray):
                x_in = torch.from_numpy(x_in).to(device=self.device, dtype=self.model_dtype)
            return self.predict_single(x_in, aggregate="all").view(-1, 1)

        def _custom_mask(mask: NDArray[np.int_], x_mask: NDArray[np.float_]) -> NDArray[np.float_]:
            mask = mask.reshape(1, -1)[:, :, np.newaxis]
            mask = mask.repeat(self.input_dim, axis=2)
            out = mask * self._x
            return out  # type: ignore[no-any-return]

        explainer = shap.Explainer(_explain_forward, _custom_mask, feature_names=feature_names, **kwargs)
        return explainer(placeholder, max_evals=max_evals, batch_size=1)


class RegularizedVAE(LogAutoEncoder):
    def __init__(
        self,
        model_name: str = "LSTMVariationalAutoEncoder",
        model_args: dict[str, Any] | None = None,
        loss: str = "MSELoss",
        compile_model: bool = True,
        compilation_mode: Literal["reduce-overhead", "max-autotune"] = "reduce-overhead",
        metrics: MetricCollection | None = None,
        optimizer: str = "AdamW",
        optimizer_config: dict[str, Any] | None = None,
        scheduler: str | None = None,
        scheduler_config: dict[str, Any] | None = None,
    ):
        """Initialize the AutoEncoder model with the specified parameters.

        Args:
            model_name (str, optional): The name of the model registered in the MODEL_REGISTRY.
                Defaults to "LSTMVariationalAutoEncoder".
            model_args (dict[str, Any], optional): Additional keyword arguments for the model.
            loss (str, optional): The loss function to use. Default is "MSELoss".
            compile_model (bool, optional): If True, the model is compiled. Default is True.
            compilation_mode (str, optional): The compilation mode to use. Default is "reduce-overhead".
            metrics (MetricCollection, optional): The metrics to use. Default is None.
            optimizer (str, optional): The optimizer to use. Default is "AdamW".
            optimizer_config (dict[str, Any], optional): Additional keyword arguments for the optimizer.
                Default is dict(lr=1e-4, weight_decay=1e-6).
            scheduler (str | None, optional): The scheduler to use. Default is None.
            scheduler_config (dict[str, Any], optional): Additional keyword arguments for the scheduler.
                Default is dict().

        """
        super().__init__(
            model_name=model_name,
            model_args=model_args,
            loss=loss,
            compile_model=compile_model,
            compilation_mode=compilation_mode,
            metrics=metrics,
            optimizer=optimizer,
            optimizer_config=optimizer_config,
            scheduler=scheduler,
            scheduler_config=scheduler_config,
        )

    def collect_reg_terms(self) -> Tensor:
        """Collect regularization terms."""
        return cast(Tensor, sum(m.reg_loss for _, m in self.ae.named_modules() if hasattr(m, "reg_loss")))

    @override
    def score(
        self,
        input: Tensor,
        target: Tensor,
        dim: int | list[int] | None = None,
        add_relu: bool = False,
        mask: Tensor | None = None,
        use_reg_loss: bool = False,
    ) -> Tensor:
        """Compute the mean prediction score for the given inputs.

        Args:
            input (Tensor): The input data.
            target (Tensor): The target data.
            add_relu (bool, optional): If True, the ReLU function is applied to the output. Default is False.
            dim (int | tuple[int, ...], optional): Dimension along which to reduce. Defaults to 0.
            mask (Tensor | None): A mask tensor for ignoring timesteps.
            use_reg_loss (bool, optional): If True, the regularization loss is added to the loss. Default is False.

        Returns:
            Tensor: The mean prediction score for each sample in the batch, with shape (B,).

        """
        score = self.loss(input, target)
        score = reduce(score, mask=mask, dim=dim)

        if add_relu:
            score = nn.functional.relu(score)

        if torch.is_grad_enabled() and dim is None:
            score += self.collect_reg_terms()
        return cast(Tensor, score)
