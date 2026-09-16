# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from torch import Tensor


def softmax_masking(scores: Tensor, threshold: float | Tensor) -> Tensor:
    """Apply a softmax masking technique to the given scores based on the specified threshold.

    This function generates masks for different ranges (hard, semi-soft, soft) based on the quantiles of the scores.
    It then performs random sampling for semi-soft and soft samples to create a combined mask.

    Args:
        scores (Tensor): A tensor containing the scores to be masked.
        threshold (float | Tensor): A float or tensor specifying the threshold for quantile calculation.
                                    If a float is provided, it will be used to create a default threshold tensor.

    Returns:
        Tensor: A tensor containing the combined mask after applying the softmax masking technique.

    """
    default_device: torch.device = scores.device

    if isinstance(threshold, float):
        default_threshold = torch.tensor([threshold, 0.94, 0.9], device=default_device)
    else:
        default_threshold = threshold

    quantiles = scores.quantile(default_threshold)

    # Generate masks for the three different ranges (hard, semi-soft, soft)
    mask_hard = scores > quantiles[0]  # All hard examples are selected without extremes
    mask_semi_soft = (scores < quantiles[0]) & (scores > quantiles[1])
    mask_soft = (scores < quantiles[1]) & (scores > quantiles[2])

    # Random sampling for semi-soft samples
    if mask_semi_soft.any():
        prob_semi_soft = mask_hard.sum() / mask_semi_soft.sum()

        bernoulli_trials_semi_soft = torch.rand(mask_semi_soft.sum().int(), device=scores.device)  # type: ignore
        selected_semi_soft = bernoulli_trials_semi_soft < prob_semi_soft
        true_indices_semi_soft = mask_semi_soft.nonzero(as_tuple=True)[0]
        mask_semi_soft[true_indices_semi_soft] = selected_semi_soft

    # Random sampling for soft samples
    if mask_soft.any():
        prob_soft = mask_hard.sum().float() / mask_soft.sum().float()

        bernoulli_trials_soft = torch.rand(mask_soft.sum().int(), device=scores.device)  # type: ignore
        selected_soft = bernoulli_trials_soft < prob_soft
        true_indices_soft = mask_soft.nonzero(as_tuple=True)[0]
        mask_soft[true_indices_soft] = selected_soft

    combined_mask = mask_hard | mask_semi_soft | mask_soft
    return combined_mask
