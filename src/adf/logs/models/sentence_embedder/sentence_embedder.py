# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import numpy as np
import os
import polars as pl
import tarfile
import torch
import torch.nn as nn

from collections.abc import Callable, Iterable
from numpy.typing import NDArray
from pathlib import Path
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import SimilarityFunction
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.model_card import SentenceTransformerModelCardData
from sentence_transformers.training_args import BatchSamplers
from torch._dynamo.eval_frame import OptimizedModule
from typing import TYPE_CHECKING, Any, Literal, cast
from typing_extensions import override

if TYPE_CHECKING:
    from datasets import Dataset

from .evaluator import EmbeddingDistanceEvaluator


def transform_sentences_to_input_examples(examples: Dataset, column: str = "EventTemplate") -> Dataset:
    """Transform the sentences in the specified column of input DataFrame into input examples for sentence embedding.

    Args:
        examples (Dataset): The input DataFrame containing the sentences.
        column (str, optional): The name of the column containing the sentences. Defaults to "EventTemplate".

    Returns:
        examples (Dataset): The input DataFrame with the sentences transformed into input examples.

    """
    examples["sentence1"] = [x for x in examples[column]]
    examples["sentence2"] = [x for x in examples[column]]

    return examples


class SimCSETransformer(SentenceTransformer):
    _setup_msg = "Please call setup() before training."

    """
    This class aims to provide a custom implementation of the SentenceTransformer class.
    """

    def __init__(
        self,
        model_name_or_path: str | None = "sentence-transformers/all-MiniLM-L6-v2",
        modules: Iterable[nn.Module] | None = None,
        device: str | None = None,
        prompts: dict[str, str] | None = None,
        default_prompt_name: str | None = None,
        similarity_fn_name: str | SimilarityFunction | None = None,
        cache_folder: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
        use_auth_token: bool | str | None = None,
        truncate_dim: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        model_card_data: SentenceTransformerModelCardData | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ):
        """Initialize the SentenceEmbedder.

        Args:
            model_name_or_path (str | None): Path to the pre-trained model or model identifier from Hugging Face Hub.
            modules (Iterable[nn.Module] | None): List of modules to be used in the model.
            device (str | None): Device to run the model on (e.g., 'cpu', 'cuda').
            prompts (dict[str, str] | None): Dictionary of prompts to be used for the model.
            default_prompt_name (str | None): Default prompt name to be used.
            similarity_fn_name (str | SimilarityFunction | None): Name of the sim function or a SimilarityFunction.
            cache_folder (str | None): Folder to cache the model.
            trust_remote_code (bool): Whether to trust remote code.
            revision (str | None): Model version to use.
            local_files_only (bool): Whether to use only local files.
            token (bool | str | None): Token for authentication.
            use_auth_token (bool | str | None): Token for authentication (deprecated).
            truncate_dim (int | None): Dimension to truncate the embeddings.
            model_kwargs (dict[str, Any] | None): Additional keyword arguments for the model.
            tokenizer_kwargs (dict[str, Any] | None): Additional keyword arguments for the tokenizer.
            config_kwargs (dict[str, Any] | None): Additional keyword arguments for the configuration.
            model_card_data (SentenceTransformerModelCardData | None): Data for the model card.
            backend (Literal["torch", "onnx", "openvino"]): Backend to use for the model.

        """
        if model_name_or_path and model_name_or_path.endswith(".gz"):
            with tarfile.open(model_name_or_path, "r") as tar:
                model_name_or_path = model_name_or_path.split(".gz")[0]
                tar.extractall(model_name_or_path)

        super().__init__(
            model_name_or_path,
            modules,
            device,
            prompts,
            default_prompt_name,
            similarity_fn_name,
            cache_folder,
            trust_remote_code,
            revision,
            local_files_only,
            token,
            use_auth_token,
            truncate_dim,
            model_kwargs,
            tokenizer_kwargs,
            config_kwargs,
            model_card_data,
            backend,
        )

    @override
    def compile(self, **compile_kwargs: Any) -> OptimizedModule:
        """Compile the model with `torch.compile`.

        Args:
            **compile_kwargs (Any): Additional keyword arguments for torch.compile.

        Returns:
            OptimizedModule: The compiled model.

        """
        m = torch.compile(self, **compile_kwargs)
        return cast(OptimizedModule, m)

    def setup(
        self,
        dataset_path: str,
        train_batch_size: int,
        evaluator_batch_size: int,
        transforms: list[Callable[[pl.DataFrame], pl.DataFrame]] | None = None,
        batched_transform: bool = True,
    ) -> None:
        """Set up the sentence embedder with the given configurations.

        Args:
            dataset_path (str): The path to the dataset file (parquet).
            train_batch_size (int): The batch size to use for training.
            evaluator_batch_size (int): The batch size to use for evaluation.
            transforms (list[Callable] | None): A list of transformation functions to apply to the dataset.
                Defaults to None.
            batched_transform (bool): Whether to apply transformations in a batched manner. Defaults to True.

        Returns:
            None

        """
        from datasets import Dataset

        train_dataset = Dataset.from_parquet(dataset_path, split="train")
        if transforms is not None:
            if not isinstance(transforms, list):
                transforms = [transforms]
            for transform in transforms:
                train_dataset = train_dataset.map(transform, batched=batched_transform, batch_size=train_batch_size)

        evaluator = EmbeddingDistanceEvaluator.from_hf_dataset(
            train_dataset, batch_size=evaluator_batch_size, main_similarity=SimilarityFunction.COSINE, write_csv=True
        )

        self.evaluator, self.train_dataset = evaluator, train_dataset

    @override
    def fit(  # type: ignore[override]
        self,
        learning_rate: float,
        warmup_ratio: float,
        num_epochs: int,
        output_dir: str,
        batch_size: int,
        eval_strategy: str = "steps",
        metric_for_best_model: str = "eval__quantile_distance",
        evaluation_steps: int = 1,
        **training_kwargs: Any,
    ) -> None:
        """Train the sentence embedder model with the specified training parameters.

        Args:
            learning_rate (float): Learning rate for the optimizer.
            warmup_ratio (float): Fraction of total training steps for linear warmup to `learning_rate`.
            num_epochs (int): Number of epochs to train the model.
            output_dir (str): Directory to save model checkpoints and outputs.
            batch_size (int): Batch size for training.
            eval_strategy (str): Evaluation strategy during training (e.g., "steps"). Defaults to "steps".
            metric_for_best_model (str): Metric used to select the best model. Defaults to "eval__quantile_distance".
            evaluation_steps (int): Number of steps between evaluations. Defaults to 1.
            **training_kwargs (Any): Additional keyword arguments for training configuration.

        Returns:
            None

        """
        if self.train_dataset is None:
            raise ValueError(self._setup_msg)

        train_loss = CachedMultipleNegativesRankingLoss(self, mini_batch_size=batch_size)

        args = SentenceTransformerTrainingArguments(
            metric_for_best_model=metric_for_best_model,
            learning_rate=learning_rate,
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            warmup_ratio=warmup_ratio,
            eval_strategy=eval_strategy,
            eval_steps=evaluation_steps,
            greater_is_better=True,
            save_only_model=True,
            load_best_model_at_end=True,
            batch_sampler=BatchSamplers.NO_DUPLICATES,  # in-batch negatives are duplicates of the anchor or positives
            report_to=["tensorboard"],
            **training_kwargs,
        )

        trainer = SentenceTransformerTrainer(
            model=self,
            args=args,
            train_dataset=self.train_dataset.select_columns(["sentence1", "sentence2"]),
            evaluator=self.evaluator,
            loss=train_loss,
        )

        trainer.train()

    def compute_min_gap(self, cluster_idx_col: str, batch_size: int) -> tuple[float, NDArray[np.float_]]:
        """Compute the minimum gap between sentence embeddings and returns the contrastive gap and cluster embeddings.

        Args:
            cluster_idx_col (str): The column name in the dataset that contains cluster indices.
            batch_size (int): The batch size to use for encoding sentences.

        Returns:
            tuple[float, dict]: A tuple containing:
                - contrastive_gap (float): The minimum gap between the embeddings.
                - cluster_embedding (dict[int, npt.NDArray[np.float64]]): A dictionary mapping cluster indices
                    to their corresponding embeddings.

        """
        if self.train_dataset is None:
            raise ValueError(self._setup_msg)

        for required in ["sentence1", "sentence2", cluster_idx_col]:
            if required not in self.train_dataset.column_names:
                raise ValueError(f"Column '{required}' not found in the dataset.")

        sentences = self.train_dataset.select_columns(["sentence1", "sentence2"]).to_list()
        cluster_inds = self.train_dataset[cluster_idx_col]

        embeddings = self.encode(
            sentences=sentences,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=False,  # uses torch.nn.functional.normalize on float32 embeddings
        )
        # Normalize embeddings
        embeddings = embeddings.to(torch.float32)
        torch.nn.functional.normalize(embeddings, p=2, dim=1, eps=0, out=embeddings)

        similarity = torch.mm(embeddings, embeddings.T)  # [-1, 1], 0: orthogonal

        ## calculating distances
        distances = 1 - similarity  # [0, 2], 0: same, 1: orthogonal, 2: opposite
        idx = torch.triu_indices(distances.size(0), distances.size(0), offset=1)
        upper_elements = distances[idx[0], idx[1]]

        quantile_distance = torch.quantile(upper_elements, q=0.1).item()

        self._cluster_embedding = np.zeros((max(cluster_inds) + 1, embeddings.shape[1]), dtype=np.float32)

        for i, cluster_idx in enumerate(cluster_inds):
            self._cluster_embedding[cluster_idx] = embeddings[i].cpu().numpy()

        return quantile_distance, self._cluster_embedding

    def optimized_gap(
        self,
        retrain_transformer: bool,
        cluster_idx_col: str,
        batch_size: int,
        min_contrastive_gap: float,
        max_iter: int,
        num_epochs: int,
        learning_rate: float,
        warmup_ratio: float,
        output_dir: str,
    ) -> tuple[float, NDArray[np.float_]]:
        """Minimize the contrastive gap between sentence embeddings.

        If `retrain_transformer` is True, this method will retrain the transformer model.
        It will iteratively compute the contrastive gap and adjust the model parameters to minimize
        this gap until it is below a specified threshold or the maximum number of iterations is reached.
        If `retrain_transformer` is False, this method will compute the minimum gap from the dataset.

        Args:
            retrain_transformer (bool): A flag indicating whether to retrain the transformer model.
            cluster_idx_col (int): The column name in the dataset that contains cluster indices.
            batch_size (int): The size of the batches for processing sentences.
            min_contrastive_gap (float): The minimum acceptable contrastive gap.
            max_iter (int): The maximum number of iterations to perform.
            num_epochs (int): The number of epochs to train the model.
            learning_rate (float): The learning rate for the optimizer.
            warmup_ratio (float): The fraction of total training steps for linear warmup to `learning_rate`.
            output_dir (str): The directory to save model checkpoints and outputs.

        Returns:
            tuple[float, dict[int, npt.NDArray[np.float64]]]: A tuple containing:
                - The final contrastive gap (float).
                - Cluster embeddings (dict[int, npt.NDArray[np.float64]]).

        """
        if retrain_transformer and max_iter > 1:
            for it in range(max_iter):
                cgap, cl_embedding = self.compute_min_gap(cluster_idx_col=cluster_idx_col, batch_size=batch_size)
                logging.info(f"Iter  {it + 1} / {max_iter} cgap: {cgap} min_cgap: {min_contrastive_gap}")
                if cgap < min_contrastive_gap:
                    self.fit(
                        learning_rate=learning_rate,
                        warmup_ratio=warmup_ratio,
                        num_epochs=num_epochs,
                        output_dir=output_dir,
                        batch_size=batch_size,
                    )
                else:
                    break
            return cgap, cl_embedding  # noqa
        else:
            return self.compute_min_gap(cluster_idx_col=cluster_idx_col, batch_size=batch_size)

    @property
    def cluster_embeddings(self) -> NDArray[np.float_]:
        """Return cluster embeddings."""
        return self._cluster_embedding

    @override
    def save_pretrained(
        self,
        path: str | Path,
        model_name: str | None = None,
        create_model_card: bool = True,
        train_datasets: list[str] | None = None,
        safe_serialization: bool = True,
    ) -> None:
        """Save the pretrained model to a specified directory.

        Args:
            path (str | Path): The directory where the model will be saved.
            model_name (str | None): Optional name for the model. If None, a default name is used. Defaults to None.
            create_model_card (bool): Whether to create a model card summarizing the model. Defaults to True.
            train_datasets (list[str] | None): List of training datasets used to train the model. Defaults to None.
            safe_serialization (bool): Whether to save the model using safe serialization. Default True.

        """
        # Save the model to the output directory
        if isinstance(path, Path):
            path = path.as_posix()
        Path(path).mkdir(parents=True, exist_ok=True)

        if model_name is not None:
            path = os.path.join(path, model_name)

        super().save_pretrained(
            path=path,
            model_name=model_name,
            create_model_card=create_model_card,
            train_datasets=train_datasets,
            safe_serialization=safe_serialization,
        )
