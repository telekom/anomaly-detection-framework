# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from lightning.pytorch.trainer import Trainer
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import override

from adf.logs.models.autoencoder.model import LogAutoEncoder

# Window scores emitted per epoch: high first, low second. Any metric state that
# survives the epoch boundary keeps the epoch-0 value visible in epoch 1.
EPOCH_SCORES = (100.0, 1.0)


class EpochPinnedScoreAutoEncoder(LogAutoEncoder):
    """LogAutoEncoder whose per-window score is pinned to a fixed value per epoch.

    Only the aggregated score fed to the metric collections is pinned; the loss
    path (``dim=None``) keeps the real value so the optimizer still has a graph.
    ``epoch_scores`` is indexed by ``current_epoch`` and clamped to its last entry.
    """

    @property
    def epoch_scores(self) -> tuple[float, ...]:
        return self.__dict__.setdefault("_epoch_scores", EPOCH_SCORES)

    @epoch_scores.setter
    def epoch_scores(self, value: tuple[float, ...]) -> None:
        self.__dict__["_epoch_scores"] = value

    @property
    def recorded_log_dicts(self) -> list[dict[str, float]]:
        return self.__dict__.setdefault("_recorded_log_dicts", [])

    @override
    def score(
        self,
        input: Tensor,
        target: Tensor,
        dim: int | list[int] | None = None,
        add_relu: bool = False,
        mask: Tensor | None = None,
    ) -> Tensor:
        score = super().score(input, target, dim=dim, add_relu=add_relu, mask=mask)
        if dim is None:
            return score
        pinned = self.epoch_scores[min(self.current_epoch, len(self.epoch_scores) - 1)]
        return torch.full_like(score, pinned)

    @override
    def log_dict(self, dictionary, *args, **kwargs):  # type: ignore[no-untyped-def, override]
        if not self.trainer.sanity_checking:
            self.recorded_log_dicts.append({key: float(value) for key, value in dictionary.items()})
        return super().log_dict(dictionary, *args, **kwargs)


def _entries(model: EpochPinnedScoreAutoEncoder, prefix: str) -> list[dict[str, float]]:
    return [entry for entry in model.recorded_log_dicts if any(key.startswith(prefix) for key in entry)]


def _trainer(**kwargs) -> Trainer:
    return Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **kwargs,
    )


@pytest.fixture
def model() -> EpochPinnedScoreAutoEncoder:
    torch.manual_seed(0)
    return EpochPinnedScoreAutoEncoder(
        model_name="LSTMAutoEncoder",
        model_args={
            "input_dim": 4,
            "hidden_dim": 2,
            "num_layers": 2,
            "dropout": 0.0,
            "enc_dropout": 0.0,
            "dec_dropout": 0.0,
            "sigma": 0.0,
        },
        compile_model=False,
        optimizer_config={"lr": 1e-4, "weight_decay": 0.0},
    )


@pytest.fixture
def dataloader() -> DataLoader:
    torch.manual_seed(0)
    return DataLoader(TensorDataset(torch.randn(8, 3, 4)), batch_size=4, shuffle=False)


def test_epoch_metrics_do_not_leak_across_epochs(model, dataloader):
    """Epoch 1 must log epoch 1's scores, not the running aggregate since epoch 0."""
    # Sanity checking runs validation before epoch 0 — its updates must not leak either.
    _trainer(max_epochs=2, num_sanity_val_steps=2).fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)

    train_entries = _entries(model, "train_")
    val_entries = _entries(model, "val_")
    assert len(train_entries) == 2
    assert len(val_entries) == 2

    for entries, prefix in ((train_entries, "train"), (val_entries, "val")):
        first, second = entries
        assert first[f"{prefix}_score_max"] == pytest.approx(EPOCH_SCORES[0])
        assert second[f"{prefix}_score_max"] == pytest.approx(EPOCH_SCORES[1])
        assert second[f"{prefix}_score_mean"] == pytest.approx(EPOCH_SCORES[1])
        assert second[f"{prefix}_score_quantile"] == pytest.approx(EPOCH_SCORES[1])


def test_test_epoch_metrics_do_not_leak_across_runs(model, dataloader):
    """A second trainer.test call must not report the first call's running maximum."""
    trainer = _trainer()

    # current_epoch stays 0 outside of fit, so swap the pinned score between runs.
    model.epoch_scores = (EPOCH_SCORES[0],)
    trainer.test(model, dataloaders=dataloader)
    first = _entries(model, "test_")[-1]

    model.epoch_scores = (EPOCH_SCORES[1],)
    trainer.test(model, dataloaders=dataloader)
    second = _entries(model, "test_")[-1]

    assert first["test_score_max"] == pytest.approx(EPOCH_SCORES[0])
    assert second["test_score_max"] == pytest.approx(EPOCH_SCORES[1])
    assert second["test_score_mean"] == pytest.approx(EPOCH_SCORES[1])
