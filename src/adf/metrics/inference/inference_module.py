# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import polars as pl
import torch

from datetime import datetime
from importlib import import_module
from omegaconf import OmegaConf
from pathlib import Path
from torch import nn
from typing import Any, cast
from typing_extensions import Self

from adf.core.dataframe.api import add_forward_sliding_window, lists_to_arrays
from adf.metrics.preprocessing.steps import MetricsNormalizer, compute_cyclic_features

from .inference_dataset import MetricsInferenceDataset
from .reconstruction_scores import pooled_and_per_feature_mse, ranked_reconstruction_mse_per_window
from .registry import add_to_metrics_registry

# Default backbone for metrics checkpoints (fully qualified import path).
_DEFAULT_MODEL_IMPORT_PATH = "adf.metrics.models.autoencoder.lstm.LSTMAutoEncoder"
# When ``model`` is a bare class name (no dots), resolve under this package (legacy / shorthand).
_LEGACY_METRICS_AE_PACKAGE = "adf.metrics.models.autoencoder.lstm"


def _load_nn_class_from_import_path(qualified_name: str) -> type[nn.Module]:
    """Load a ``torch.nn.Module`` subclass from an adf import path.

    Expects ``package.submodule.ClassName``, e.g.
    ``adf.metrics.models.autoencoder.lstm.LSTMAutoEncoder`` or
    ``adf.logs.models.autoencoder.lstm.LSTMAutoEncoder`` so the same pattern
    works for metrics and logs autoencoder backbones.

    A bare ``ClassName`` (no dots) is treated as a class inside
    ``adf.metrics.models.autoencoder.lstm`` for backward compatibility.

    Args:
        qualified_name: Full dotted import path ending with the class name.

    Returns:
        The resolved class type.

    """
    name = qualified_name.strip()
    if not name:
        raise ValueError("model import path in model_config.json must be a non-empty string.")
    if "." not in name:
        package, attr = _LEGACY_METRICS_AE_PACKAGE, name
    else:
        package, attr = name.rsplit(".", 1)
        if not package or not attr:
            raise ValueError(
                f"Invalid model import path {qualified_name!r}. "
                "Use 'package.module.ClassName' (e.g. adf.metrics.models.autoencoder.lstm.LSTMAutoEncoder)."
            )
    try:
        mod = import_module(package)
    except ImportError as e:
        raise ValueError(f"Cannot import package {package!r} for model path {qualified_name!r}.") from e
    try:
        cls = getattr(mod, attr)
    except AttributeError as e:
        raise ValueError(f"Package {package!r} has no attribute {attr!r} (model path: {qualified_name!r}).") from e
    if not isinstance(cls, type) or not issubclass(cls, nn.Module):
        raise TypeError(f"{qualified_name!r} does not resolve to a torch.nn.Module subclass.")
    return cls


def _checkpoint_hparams_as_dict(raw: Any) -> dict[str, Any]:
    """Turn checkpoint ``hyperparameters`` / ``hparams`` into a plain ``dict``."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if OmegaConf.is_config(raw):
        out = OmegaConf.to_container(raw, resolve=True)
        if isinstance(out, dict):
            return cast(dict[str, Any], out)
        return {}
    try:
        return dict(raw)
    except Exception:
        pass
    try:
        return dict(vars(raw))
    except Exception:
        pass
    return {}


# When hyperparameters have no ``model_args`` (``MetricsAutoEncoder`` checkpoints), start from
# these defaults and overlay matching root keys (same names as ``LSTMAutoEncoder`` kwargs).
_DEFAULT_METRICS_LSTM_MODEL_ARGS: dict[str, Any] = {
    "input_dim": 14,
    "hidden_dim": 2,
    "time_dim": 6,
    "num_layers": 2,
    "dropout": 0.0,
    "sigma": 0.0,
    "batch_first": True,
    "return_hidden": False,
}
# Root hyperparameter keys copied onto ``model_args`` for metrics checkpoints (no nested ``model_args``).
_METRICS_HPARAM_KEYS_FOR_LSTM: frozenset[str] = frozenset(
    {
        "input_dim",
        "hidden_dim",
        "num_layers",
        "dropout",
        "enc_dropout",
        "dec_dropout",
        "sigma",
        "batch_first",
        "return_hidden",
        "time_dim",
    }
)


def _model_args_from_hparams(hparams_dict: dict[str, Any]) -> dict[str, Any]:
    """Build ``model_args`` for ``model_cls(**model_args)``.

    If ``model_args`` is set and non-empty on the checkpoint, use that dict only. Otherwise use
    ``_DEFAULT_METRICS_LSTM_MODEL_ARGS`` and overlay any matching root keys (metrics training).
    """
    nested = hparams_dict.get("model_args")
    if nested is not None:
        subtree = _checkpoint_hparams_as_dict(nested)
        if subtree:
            return subtree
    out = dict(_DEFAULT_METRICS_LSTM_MODEL_ARGS)
    for k, v in hparams_dict.items():
        if k in _METRICS_HPARAM_KEYS_FOR_LSTM:
            out[k] = v
    return out


def _strip_torch_compile_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove ``._orig_mod`` from state dict keys (compiled ``nn.Module`` / ``_orig_mod`` nodes).

    Same normalization as ``dbaas.tasks.utilities.fix_state_dict``.
    """
    return {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}


@add_to_metrics_registry
class MetricsInferencePipeline:
    def __init__(
        self,
        model: nn.Module,
        threshold: float,
        *,
        normalizer: MetricsNormalizer,
        column_order: list[str] | None = None,
        window_size: int = 10,
        input_dim: int = 14,
        time_dim: int = 6,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize the MetricsInference module.

        Args:
            model (nn.Module): Trained metrics autoencoder (e.g. LSTMAutoEncoder)
            threshold (float): Anomaly threshold for reconstruction error
            normalizer (MetricsNormalizer): Fitted normalizer (StandardScaler, RobustScaler, or custom).
            column_order (list[str], optional): Column order for model input. Excludes timestamp.
            window_size (int, optional): Size of sliding window. Default is 10.
            input_dim (int, optional): Number of input dimensions (metrics + time features). Default is 14.
            time_dim (int, optional): Number of time-related features. Default is 6.
            device (str | torch.device, optional): Device to run inference on. Default is None.

        """
        self.window_size = window_size
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalizer = normalizer
        self.column_order = column_order
        self.threshold = threshold

        # Set up model
        self.model = model
        self.tensor_type = self.model.dtype

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        artifact_dir: str | Path,
        *,
        window_size: int | None = None,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        time_dim: int | None = None,
        num_layers: int | None = None,
        return_latent: bool = False,
    ) -> Self:
        """Create an inference instance from metrics artifact directory.

        Expects MetricsNormalizer format: scalars.joblib (or scalers.joblib), anomaly_threshold.json,
        columns.json. Supports StandardScaler, RobustScaler, and custom (arcsinh+RobustScaler).
        Same format as Tardis and future metrics use cases using MetricsDataProcessor for training.

        Model backbone is built as ``model_cls(**model_args)``. If the checkpoint has a
        ``model_args`` dict (e.g. ``LogAutoEncoder``), it is used as-is. Otherwise defaults for
        metrics ``LSTMAutoEncoder`` are applied and any matching root hyperparameters are copied
        over (``MetricsAutoEncoder``). Optional ``from_pretrained`` arguments override when not
        ``None``.

        ``model_config.json`` may set ``window_size`` and ``model`` (fully
        qualified import path of the backbone ``nn.Module``, e.g.
        ``adf.metrics.models.autoencoder.lstm.LSTMAutoEncoder`` or
        ``adf.logs.models.autoencoder.lstm.LSTMAutoEncoder``). A bare class name is resolved
        under ``adf.metrics.models.autoencoder.lstm``. Default is the metrics LSTM path above.
        ``window_size`` can still be overridden by the passed argument when not None.

        Args:
            artifact_dir: Directory containing model artifacts.
            window_size: Override for sliding window size. If None, read from model_config.json
                in artifact when present, else default 10.
            input_dim: If set, overrides checkpoint / subtree values.
            hidden_dim: If set, overrides checkpoint / subtree values.
            time_dim: If set, overrides checkpoint (metrics LSTM / pipeline metadata). If ``None``,
                use value from kwargs or root ``hyperparameters``, else 6.
            num_layers: If set, overrides checkpoint. If ``None``, use checkpoint or class default.
            return_latent: If True, the latent vector is returned. Defaults to False.

        Returns:
            Configured MetricsInferencePipeline instance.

        """
        artifact_dir = Path(artifact_dir)

        model_config: dict[str, Any] = {}
        model_config_path = artifact_dir / "model_config.json"
        if model_config_path.exists():
            with open(model_config_path) as f:
                model_config = json.load(f)

        if window_size is None:
            window_size = model_config.get("window_size", 10)

        model_import_path = model_config.get("model", _DEFAULT_MODEL_IMPORT_PATH)
        model_cls = _load_nn_class_from_import_path(model_import_path)

        # MetricsNormalizer format (scalars.joblib, anomaly_threshold.json, columns.json)
        for name in ("model.pth", "autoencoder_model.pth", "autoencoder_finetuned.pth"):
            model_path = artifact_dir / name
            if model_path.exists():
                break
        else:
            raise FileNotFoundError(
                f"No model.pth, autoencoder_model.pth or autoencoder_finetuned.pth in {artifact_dir}"
            )

        # Load checkpoint and extract state dict + hyperparameters
        loaded = torch.load(model_path, weights_only=False, map_location="cpu")
        state_dict = None
        hparams = {}

        if isinstance(loaded, dict):
            if "model_state_dict" in loaded:
                state_dict = loaded["model_state_dict"]
                hparams = loaded.get("hyperparameters", loaded.get("hparams", {}))
            elif "state_dict" in loaded:
                state_dict = loaded["state_dict"]
                hparams = loaded.get("hyperparameters", loaded.get("hparams", {}))
            else:
                raise ValueError(f"Unknown checkpoint format. Keys: {list(loaded.keys())}")
        else:
            raise ValueError("Expected checkpoint dict, got direct model object")

        if state_dict:
            state_dict = _strip_torch_compile_state_dict_keys(state_dict)

        # Strip 'ae.' prefix if present
        if state_dict and any(k.startswith("ae.") for k in state_dict):
            state_dict = {k.replace("ae.", "", 1): v for k, v in state_dict.items()}

        hparams_dict = _checkpoint_hparams_as_dict(hparams)
        model_args = _model_args_from_hparams(hparams_dict)
        if input_dim is not None:
            model_args["input_dim"] = input_dim
        if hidden_dim is not None:
            model_args["hidden_dim"] = hidden_dim
        if time_dim is not None:
            model_args["time_dim"] = time_dim
        if num_layers is not None:
            model_args["num_layers"] = num_layers

        pipe_input_dim = int(model_args.get("input_dim", hparams_dict.get("input_dim", 14)))
        pipe_time_dim = int(model_args.get("time_dim", hparams_dict.get("time_dim", 6)))

        model = model_cls(**model_args)
        model.load_state_dict(state_dict, strict=True)

        if hasattr(model, "return_latent") and return_latent:
            model.return_latent = return_latent  # type: ignore[assignment]
        else:
            logging.warning(f"Model {model_cls.__name__} does not have a return_latent attribute")

        # Load scalars
        scalars_path = artifact_dir / "scalars.joblib"
        if not scalars_path.exists():
            scalars_path = artifact_dir / "scalers.joblib"
        if not scalars_path.exists():
            raise FileNotFoundError(f"No scalars.joblib in {artifact_dir}")
        normalizer = MetricsNormalizer.load_scalers(scalars_path)

        # Load threshold
        threshold_path = artifact_dir / "anomaly_threshold.json"
        if not threshold_path.exists():
            raise FileNotFoundError(f"No anomaly_threshold.json in {artifact_dir}")
        with open(threshold_path) as f:
            threshold_data = json.load(f)
        threshold = threshold_data["threshold"]

        # Load column order
        columns_path = artifact_dir / "columns.json"
        if not columns_path.exists():
            raise FileNotFoundError(f"No columns.json in {artifact_dir}")
        with open(columns_path) as f:
            columns = json.load(f)
        column_order = [c for c in columns if c not in ("timestamp", "index")]

        return cls(
            model=model,
            threshold=threshold,
            normalizer=normalizer,
            column_order=column_order,
            window_size=window_size,
            input_dim=pipe_input_dim,
            time_dim=pipe_time_dim,
        )

    def _create_inf_dataset(
        self, data: pl.DataFrame, timestamps: str, zero_padding: bool = True
    ) -> tuple[torch.Tensor, list[datetime]]:
        """Create sliding windows from input data with zero padding for short sequences.

        Args:
            data (pl.DataFrame): Input data (feature columns only, no timestamp)
            timestamps (str): Corresponding timestamps column
            zero_padding (bool, optional): Whether to zero pad short sequences. Default is True.

        Returns:
            Tuple of windowed data and corresponding timestamps.
            If data length < window_size, returns single zero-padded window.

        """
        tstamps = data.select(timestamps)
        data = data.drop(timestamps)

        if data.is_empty():
            raise pl.exceptions.NoDataError("Input data is empty")

        if len(data) < self.window_size:
            if not zero_padding:
                raise ValueError(
                    f"Data length {len(data)} is less than\
                    window size {self.window_size}. Cannot create windows."
                )

            logging.info(
                f"Data length {len(data)} is less than\
                window size {self.window_size}. Using zero padding."
            )
            padded_window = torch.zeros((self.window_size, data.shape[1]), dtype=self.tensor_type)
            padded_window[: len(data)] = torch.tensor(data.to_numpy())
            return padded_window.unsqueeze(0), tstamps[-1].item()

        # Replace features with windowed data
        windows_data = add_forward_sliding_window(
            data,
            column=data.schema.names(),
            window_size=self.window_size,
            output=data.schema.names(),
            filter_valid_rows=True,
        )

        windows_data = lists_to_arrays(windows_data, data.schema.names(), self.window_size)
        # add_forward_sliding_window produces (row, col) where each col is list[window_size];
        # .rows() yields (N, num_features, window_size) but model expects (N, window_size, num_features)
        windows_tensor = torch.tensor(windows_data.rows())
        if windows_tensor.dim() == 3:
            windows_tensor = windows_tensor.permute(0, 2, 1)  # (N, F, S) -> (N, S, F)
        return windows_tensor, tstamps[self.window_size - 1 :].get_column(timestamps).to_list()

    def infer(
        self,
        df: pl.DataFrame,
        normalize_columns: list[str] | None = None,
        timestamp_column: str | None = None,
        dataloader_kwargs: dict[str, Any] | None = None,
        min_sequence_length: int | None = None,
        *,
        include_granular_mse: bool = True,
        top_ranked_metrics: int = 10,
    ) -> dict[str, Any]:
        """Perform inference on input dataframe.

        Uses normalization type from artifact (StandardScaler, RobustScaler, or custom).
        Must match preprocessing/training; Tardis uses custom (arcsinh+RobustScaler).

        Args:
            df (pl.DataFrame): Input dataframe with metrics data
            normalize_columns (list[str], optional): Columns to normalize. Inferred from
                normalizer.scalers if not provided.
            timestamp_column (str, optional): Column name for timestamps. Default is None.
            dataloader_kwargs (dict[str, Any], optional): Additional arguments for DataLoader. Default is None.
            min_sequence_length (int, optional): Minimum sequence length for inference. Default is None.
            include_granular_mse: If True, also return per-metric tensors and ranked payloads.
            top_ranked_metrics: Max ranks per window in ``reconstruction_mse_per_metric`` (default 10).

        Returns:
            Dictionary containing:
                - timestamp (list[datetime]): Timestamps for each window
                - reconstruction (Tensor): Pooled reconstruction MSE per window (mean over time and metric dims)
                - anomalies (Tensor): Anomaly flags for each window (error > threshold)
                - metric_names (list[str], optional): Model metric feature names in channel order (same order used to
                  build ``reconstruction_mse_per_metric``) when ``include_granular_mse`` is True
                - reconstruction_mse_per_metric (list[dict], optional): One flat dict per window (same order as
                  ``timestamp``) with ``@timestamp``, ``rank_k_feature_name``, ``rank_k_mse``,
                  ``rank_k_contribution_percent`` for ``k = 1 .. min(top_ranked_metrics, n_metrics)`` when
                  ``include_granular_mse`` is True. Full per-metric MSE tensors are not returned.

        """
        return_latent = True if hasattr(self.model, "return_latent") and self.model.return_latent else False
        logger = logging.getLogger(__name__)
        if min_sequence_length and len(df) < min_sequence_length:
            raise ValueError(f"Input sequence length {len(df)} is less than minimum required {min_sequence_length}")

        dataloader_kwargs = dataloader_kwargs or {"batch_size": 512, "num_workers": 4, "pin_memory": True}
        timestamp_column = timestamp_column or df.schema.names()[0]

        # NaN/null handling: replace with 0 and warn
        feature_cols = [c for c in df.columns if c != timestamp_column]
        if feature_cols:
            cols_to_check = [c for c in feature_cols if c in df.columns]
            nan_or_null_per_col = df.select(
                [(pl.col(c).is_nan().sum() + pl.col(c).is_null().sum()).alias(c) for c in cols_to_check]
            )
            total_nan_null = sum(nan_or_null_per_col.row(0)) if nan_or_null_per_col.height > 0 else 0
            if total_nan_null > 0:
                cols_with_nan_null = [c for c in nan_or_null_per_col.columns if nan_or_null_per_col[c][0] > 0]
                logger.warning(
                    "NaN/null values detected: %d total. Replacing with 0.0. Columns: %s",
                    total_nan_null,
                    cols_with_nan_null,
                )
                repl_exprs = [
                    pl.when(pl.col(c).is_nan() | pl.col(c).is_null()).then(0.0).otherwise(pl.col(c)).alias(c)
                    for c in feature_cols
                    if c in df.columns
                ]
                if repl_exprs:
                    df = df.with_columns(repl_exprs)

        # Cyclic features + normalize
        metric_columns = normalize_columns or list(self.normalizer.scalers.keys())
        df = compute_cyclic_features(df, timestamp_column)
        df = self.normalizer.transform(df, metric_columns)

        # Enforce column order from columns.json
        cyclic_names = ["hour_sin", "hour_cos", "day_sin", "day_cos", "week_sin", "week_cos"]
        if self.column_order:
            model_cols = [c for c in self.column_order if c in df.columns and c != timestamp_column]
            for cn in cyclic_names:
                if cn in df.columns and cn not in model_cols:
                    model_cols.append(cn)
        else:
            expected = set(metric_columns) | set(cyclic_names)
            model_cols = [c for c in df.schema.names() if c != timestamp_column and c in expected]
        df = df.select([timestamp_column] + model_cols)

        metric_feature_names = [c for c in model_cols if c not in cyclic_names]
        n_metrics_expected = getattr(self.model, "n_metrics", None)
        if n_metrics_expected is not None and len(metric_feature_names) != n_metrics_expected:
            raise ValueError(
                f"Metric column count {len(metric_feature_names)} from dataframe/artifact order "
                f"does not match model.n_metrics={n_metrics_expected}. Check columns.json vs scalers."
            )

        windows, timestamps = self._create_inf_dataset(df, timestamp_column)
        dataset = MetricsInferenceDataset(windows)
        if len(windows) == 1:
            dataloader_kwargs = dict(dataloader_kwargs)
            dataloader_kwargs["batch_size"] = 1
        dataloader = dataset.as_dataloader(**dataloader_kwargs)

        error_tensors: list[torch.Tensor] = []
        per_metric_tensors: list[torch.Tensor] = []
        latent_tensors: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                if return_latent:
                    outputs, latent = self.model(batch)
                else:
                    outputs = self.model(batch)
                if isinstance(outputs, tuple):
                    # Metrics ``LSTMAutoEncoder``: (reconstruction, metrics_target); optional 3rd = hidden.
                    x_hat, x_metrics = outputs[0], outputs[1]
                elif outputs.shape == batch.shape:
                    # Logs ``LSTMAutoEncoder`` (dbaas deployment): full-dimension reconstruction vs input.
                    x_hat, x_metrics = outputs[..., : -self.time_dim], batch[..., : -self.time_dim]
                else:
                    raise ValueError(f"Unexpected outputs shape {outputs.shape} for batch shape {batch.shape}")
                per_f, pooled = pooled_and_per_feature_mse(x_metrics, x_hat)
                error_tensors.append(pooled)
                if return_latent:
                    latent_tensors.append(latent.cpu())
                if include_granular_mse:
                    per_metric_tensors.append(per_f)

        reconstruction_errors_tensor = torch.cat(error_tensors)
        anomalies = reconstruction_errors_tensor > self.threshold

        out: dict[str, Any] = {
            "timestamp": timestamps,
            "reconstruction": reconstruction_errors_tensor,
            "anomalies": anomalies,
        }
        if return_latent:
            out["latent"] = torch.cat(latent_tensors)
        else:
            out["latent"] = None

        if include_granular_mse:  # if explainability is required
            reconstruction_per_metric = torch.cat(per_metric_tensors)
            n_feat = reconstruction_per_metric.shape[1]
            if len(metric_feature_names) < n_feat:
                raise ValueError(f"Need at least {n_feat} metric feature names; got {len(metric_feature_names)}")
            names_out = metric_feature_names[:n_feat]
            if reconstruction_per_metric.shape[1] != len(names_out):
                raise ValueError(
                    f"Per-metric MSE width {reconstruction_per_metric.shape[1]} "
                    f"does not match metric name count {len(names_out)}"
                )
            out["metric_names"] = list(names_out)
            out["reconstruction_mse_per_metric"] = ranked_reconstruction_mse_per_window(
                reconstruction_per_metric, timestamps, list(names_out), top_k=top_ranked_metrics
            )
        return out
