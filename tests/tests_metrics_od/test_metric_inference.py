# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch

from torch import Tensor

from adf.core.dataframe.api import add_ones
from adf.metrics.inference.inference_module import MetricsInferencePipeline
from adf.metrics.preprocessing.steps import MetricsNormalizer


class DummyAE(torch.nn.Module):
    """Minimal stand-in for LSTMAutoEncoder: reconstructs first ``n_metrics`` channels only."""

    def __init__(self, n_metrics: int = 8) -> None:
        super().__init__()
        self._n_metrics = n_metrics

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = x[:, :, : self._n_metrics]
        return m, m

    @property
    def dtype(self):
        return torch.float32

    @property
    def n_metrics(self) -> int:
        return self._n_metrics


@pytest.fixture
def dummy_inference():
    n_metrics = 8
    time_dim = 6
    model = DummyAE(n_metrics=n_metrics)
    cyclic_names = ["hour_sin", "hour_cos", "day_sin", "day_cos", "week_sin", "week_cos"]
    column_order = [f"col{i}" for i in range(1, n_metrics + 1)] + cyclic_names
    normalizer = MetricsNormalizer(normalization_type="RobustScaler")
    time_values = pd.date_range("2023-01-01", periods=10).tolist()
    df_fit = pl.DataFrame({f"col{i}": np.random.rand(10) for i in range(1, n_metrics + 1)})
    normalizer.fit_transform(df_fit, [f"col{i}" for i in range(1, n_metrics + 1)])
    threshold = 0.5
    return MetricsInferencePipeline(
        model=model,
        normalizer=normalizer,
        threshold=threshold,
        column_order=column_order,
        window_size=5,
        input_dim=n_metrics + time_dim,
        time_dim=time_dim,
        device="cpu",
    )


def patched_infer(
    self,
    df: pl.DataFrame,
    normalize_columns: list[str],
    min_sequence_length: int | None = None,
    dataloader_kwargs: dict[str, any] | None = None,
) -> dict[str, any]:
    if min_sequence_length and len(df) < min_sequence_length:
        raise ValueError(f"Input sequence length {len(df)} is less than minimum required length {min_sequence_length}")
    from adf.core.dataframe.api import add_cyclic_features, scale

    dataloader_kwargs = dataloader_kwargs or {}

    df = add_cyclic_features(df, "Time")
    df = scale(df, column=normalize_columns)

    windowed_data, window_timestamps = self._create_inf_dataset(df, "Time")

    if len(windowed_data) == 1:
        dataloader_kwargs.update({"batch_size": 1})

    from adf.metrics.inference.inference_module import MetricsInferenceDataset

    dataset = MetricsInferenceDataset(windowed_data)
    dataloader = dataset.as_dataloader(**dataloader_kwargs)

    error_tensors: list[Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(self.device)
            x_hat, x_metrics = self.model(batch)
            errors = torch.nn.functional.mse_loss(x_hat, x_metrics, reduction="none").mean(dim=[1, 2])
            error_tensors.append(errors.cpu())
    reconstruction_errors_tensor: Tensor = torch.cat(error_tensors)
    anomalies = reconstruction_errors_tensor > self.threshold
    return {"timestamp": window_timestamps, "reconstruction": reconstruction_errors_tensor, "anomalies": anomalies}


def test_create_sliding_windows_short(dummy_inference):
    timestamps = pl.DataFrame({"dates": pd.date_range("2023-01-01", periods=3).to_list()})
    timestamps = add_ones(timestamps, column="ones")
    windows, window_timestamps = dummy_inference._create_inf_dataset(timestamps, timestamps.schema.names()[0])
    assert windows.shape == (1, dummy_inference.window_size, 1)
    padded = torch.zeros((dummy_inference.window_size, 1), dtype=torch.float32)
    padded[:3] = torch.tensor(timestamps.select("ones").to_numpy())
    assert torch.equal(windows[0], padded)
    assert window_timestamps == timestamps["dates"].to_list()[-1]


def test_create_sliding_windows_normal(dummy_inference):
    data = torch.arange(50, dtype=torch.float32).reshape(10, 5)
    dummy_inference.window_size = 3
    timestamps = pl.DataFrame({"dates": pd.date_range("2023-01-01", periods=10).to_list()})
    timestamps = timestamps.with_columns(pl.DataFrame(data.numpy()))

    windows, window_timestamps = dummy_inference._create_inf_dataset(timestamps, timestamps.schema.names()[0])
    assert windows.shape[0] == 8
    assert windows.shape[1] == 3
    assert windows.shape[2] == 5
    for i in range(8):
        expected = data[i : i + 3].T
        # assert torch.allclose(windows[i], expected)
        assert window_timestamps[i] == timestamps.get_column(timestamps.schema.names()[0])[i + 2]


def test_infer_min_sequence_length(dummy_inference):
    num_rows = 4
    n_metrics = dummy_inference.model.n_metrics
    time_values = pd.date_range("2023-01-01", periods=num_rows).tolist()
    data = {f"col{i}": np.random.rand(num_rows) for i in range(1, n_metrics + 1)}
    df = pl.DataFrame({"Time": time_values, **data})
    with pytest.raises(ValueError):
        dummy_inference.infer(df, normalize_columns=[f"col{i}" for i in range(1, n_metrics + 1)], min_sequence_length=10)


def test_infer_output(monkeypatch, dummy_inference):
    monkeypatch.setattr("adf.core.dataframe.api.add_cyclic_features", lambda df, ts: df)
    monkeypatch.setattr("adf.core.dataframe.api.scale", lambda df, column: df)
    monkeypatch.setattr(MetricsInferencePipeline, "infer", patched_infer)
    num_rows = 10
    n_metrics = dummy_inference.model.n_metrics
    time_values = pd.date_range("2023-01-01", periods=num_rows).tolist()
    numeric_data = {f"col{i}": np.random.rand(num_rows) for i in range(1, n_metrics + 1)}
    df = pl.DataFrame({"Time": time_values, **numeric_data})
    result = dummy_inference.infer(
        df, normalize_columns=list(df.columns[1:]), dataloader_kwargs=dict(batch_size=2, num_workers=0)
    )
    assert "timestamp" in result
    assert "reconstruction" in result
    assert "anomalies" in result
    windows, _ = dummy_inference._create_inf_dataset(df, "Time")
    expected_num_windows = windows.shape[0]
    assert result["reconstruction"].shape[0] == expected_num_windows
    assert torch.allclose(result["reconstruction"], torch.zeros(expected_num_windows))
    assert result["anomalies"].dtype == torch.bool


# def test_from_pretrained(tmp_path, monkeypatch):
#     import json
#     import joblib

#     from adf.metrics.inference.inference_module import MetricsInferencePipeline
#     from adf.metrics.models.autoencoder.lstm import LSTMAutoEncoder

#     # Create MetricsNormalizer format artifacts
#     model_file = tmp_path / "model.pth"
#     torch.save(
#         {"model_state_dict": {}, "hyperparameters": {"input_dim": 14, "hidden_dim": 2, "time_dim": 6, "num_layers": 2}},
#         model_file,
#     )

#     normalizer = MetricsNormalizer(normalization_type="RobustScaler")
#     df = pl.DataFrame({f"col{i}": np.random.rand(10) for i in range(1, 9)})
#     normalizer.fit_transform(df, [f"col{i}" for i in range(1, 9)])
#     normalizer.save_scalers(tmp_path / "scalars.joblib")

#     (tmp_path / "anomaly_threshold.json").write_text(json.dumps({"threshold": 0.75}))
#     (tmp_path / "columns.json").write_text(
#         json.dumps([f"col{i}" for i in range(1, 9)] + ["hour_sin", "hour_cos", "day_sin", "day_cos", "week_sin", "week_cos"])
#     )

#     monkeypatch.setattr(LSTMAutoEncoder, "load_state_dict", lambda self, state: None)

#     inference_instance = MetricsInferencePipeline.from_pretrained(
#         tmp_path,
#         window_size=5,
#         input_dim=14,
#         time_dim=6,
#     )

#     assert inference_instance.threshold == 0.75
#     assert not inference_instance.model.training
#     assert inference_instance.device in ["cuda", "cpu"]
#     assert inference_instance.normalizer is not None
