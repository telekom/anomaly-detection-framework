# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import lightning as L
import logging
import torch

from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from typing import Any
from typing_extensions import override


def create_weighted_sampler(scores: Tensor, quantile_threshold: float = 0.999) -> WeightedRandomSampler:
    """Create a sampler that weights training examples based on their reconstruction scores.

    This is important for anomaly detection as it helps the model focus on:
    1. Hard-to-reconstruct examples (potential anomalies)
    2. Examples with high reconstruction error

    The sampler:
    - Takes the reconstruction scores
    - Zeros out extreme outliers above the quantile threshold
    - Uses these scores as sampling weights
    - Samples with replacement, giving higher probability to
      examples with higher reconstruction error

    Args:
        scores: Reconstruction errors for each time window
        quantile_threshold: Cut-off for extreme outliers (default 0.999)

    Returns:
        Sampler that can be used with DataLoader to bias training
        towards harder examples

    """
    weights = scores.clone()
    threshold = torch.quantile(weights, quantile_threshold)
    weights[weights > threshold] = 0

    return WeightedRandomSampler(weights=weights.tolist(), num_samples=len(weights), replacement=True)


class MetricsDataModule(L.LightningDataModule):
    hparams: dict[str, Any]  # Type hint for hparams attribute
    train_dataset: TensorDataset
    val_dataset: TensorDataset
    test_dataset: TensorDataset
    scores: Tensor | None = None

    def __init__(
        self,
        dataset: Tensor,
        dataset_val: Tensor,
        batch_size: int = 32,
        num_workers: int = 4,
        threshold_perc: float = 0.999,
        use_weighted_sampling: bool = False,
    ):
        """Initialize the Metrics DataModule.

        Args:
            dataset (torch.Tensor): Input tensor with shape [num_samples, sequence_length, features]
            dataset_val (torch.Tensor): Validation tensor with shape [num_samples, sequence_length, features]
            batch_size (int): Batch size for dataloaders
            num_workers (int): Number of workers for dataloaders
            threshold_perc (float): Threshold percentage for weighted sampling
            use_weighted_sampling (bool): Whether to use weighted sampling based on reconstruction scores

        """
        super().__init__()
        self.save_hyperparameters(ignore=["dataset", "dataset_val"])

        self.dataset = dataset
        self.dataset_val = dataset_val
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.threshold_perc = threshold_perc
        self.use_weighted_sampling = use_weighted_sampling

        self.scores = None

    @override
    def setup(self, stage: str | None = None) -> None:
        """Set up the datasets for training, validation, and testing.

        Args:
            stage (str, optional): Current stage ('fit', 'validate', 'test', or 'predict')

        """
        if stage == "fit" or stage is None:
            self.train_dataset = TensorDataset(self.dataset)
            self.val_dataset = TensorDataset(self.dataset_val)

        if stage == "test":
            self.test_dataset = TensorDataset(self.dataset)

    @property
    def sampler(self) -> WeightedRandomSampler | None:
        """Return a sampler for training based on the configuration.

        If `use_weighted_sampling` is enabled in `hparams` and `scores` is not None,
        a weighted sampler is created using the provided scores and threshold percentage.
        Otherwise, a random sampler is used.

        Returns:
            Sampler or None: A weighted sampler if `use_weighted_sampling` is True and `scores` is provided,
                             otherwise None indicating random sampling.

        """
        use_weighted_sampling = self.hparams.get("use_weighted_sampling", False)
        threshold_perc = self.hparams.get("threshold_perc", 0.999)

        if use_weighted_sampling and self.scores is not None:
            logging.debug("Using weighted sampling for training")
            return create_weighted_sampler(self.scores, threshold_perc)

        logging.debug("Using random sampling for training")
        return None

    @override
    def train_dataloader(self) -> DataLoader[tuple[Tensor, ...]]:
        """Return the DataLoader for the training dataset.

        This DataLoader is configured with the training dataset, batch size,
        number of workers, sampler, and pin memory settings specified in
        the hyperparameters.

        Returns:
            train_dataloader (DataLoader): A DataLoader instance for the training dataset.

        """
        batch_size = self.hparams.get("batch_size", 32)
        num_workers = self.hparams.get("num_workers", 4)

        return DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=self.sampler, pin_memory=True
        )

    @override
    def val_dataloader(self) -> DataLoader[tuple[Tensor, ...]]:
        """Return the DataLoader for the validation dataset.

        This DataLoader is configured with the training dataset, batch size,
        number of workers, sampler, and pin memory settings specified in
        the hyperparameters.

        Returns:
            val_dataloader (DataLoader): A DataLoader instance for the training dataset.

        """
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True
        )

    @override
    def test_dataloader(self) -> DataLoader[tuple[Tensor, ...]]:
        """Return the DataLoader for the testing dataset.

        This DataLoader is configured with the training dataset, batch size,
        number of workers, sampler, and pin memory settings specified in
        the hyperparameters.

        Returns:
            test_dataloader (DataLoader): A DataLoader instance for the training dataset.

        """
        batch_size = self.hparams.get("batch_size", 32)
        num_workers = self.hparams.get("num_workers", 4)

        return DataLoader(
            self.test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True
        )
