# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import torch

from adf.metrics.inference.reconstruction_scores import (
    pooled_and_per_feature_mse,
    ranked_reconstruction_mse_per_window,
)


def test_pooled_matches_mse_loss_mean12():
    torch.manual_seed(0)
    target = torch.randn(4, 7, 5)
    recon = torch.randn(4, 7, 5)
    per_f, pooled = pooled_and_per_feature_mse(target, recon)
    expected_pooled = torch.nn.functional.mse_loss(recon, target, reduction="none").mean(dim=(1, 2))
    assert per_f.shape == (4, 5)
    assert pooled.shape == (4,)
    assert torch.allclose(pooled, expected_pooled)
    assert torch.allclose(pooled, per_f.mean(dim=1))


def test_ranked_reconstruction_mse_known_ordering():
    mse = torch.tensor([[0.10, 0.30, 0.60]])  # rank: c, b, a by mse
    ts = [datetime(2026, 6, 3, 7, 55, 0, tzinfo=timezone.utc)]
    names = ["a", "b", "c"]
    out = ranked_reconstruction_mse_per_window(mse, ts, names, top_k=10)
    assert len(out) == 1
    d = out[0]
    assert d["@timestamp"].endswith("Z")
    assert d["rank_1_feature_name"] == "c" and abs(d["rank_1_mse"] - 0.6) < 1e-6
    assert d["rank_2_feature_name"] == "b"
    assert d["rank_3_feature_name"] == "a"
    s = d["rank_1_contribution_percent"] + d["rank_2_contribution_percent"] + d["rank_3_contribution_percent"]
    assert abs(s - 100.0) < 1e-4


def test_pooled_per_feature_matches_shell_reductions():
    """Same as Shell RootCauseAnalyzer.compute_granular_mse (mean dim=1 vs dim=(1,2))."""
    original = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    reconstructed = torch.zeros((1, 2, 2))
    squared = (original - reconstructed) ** 2
    expected_per = squared.mean(dim=1)
    expected_total = squared.mean(dim=(1, 2))
    per_f, pooled = pooled_and_per_feature_mse(original, reconstructed)
    assert torch.allclose(per_f, expected_per)
    assert torch.allclose(pooled, expected_total)
