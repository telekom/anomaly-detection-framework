# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: S101

from __future__ import annotations

import logging
import torch

from contextlib import nullcontext
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.evaluation import SimilarityFunction
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import Self, override

if TYPE_CHECKING:
    from datasets import Dataset


class EmbeddingDistanceEvaluator(SentenceEvaluator):
    """Evaluate a model based on the mean and median pairwise distance between the embeddings."""

    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        batch_size: int = 16,
        main_similarity: str | SimilarityFunction | None = None,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = False,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] | None = None,
        truncate_dim: int | None = None,
    ):
        """Construct an evaluator based for the dataset.

        Args:
            sentences1 (list[str]): list with the first sentence in a pair.
            sentences2 (list[str]): list with the second sentence in a pair.
            batch_size (int, optional): The batch size for processing the sentences. Defaults to 16.
            main_similarity (str | SimilarityFunction | None): The main similarity function to use.
                Can be a string (e.g. "cosine", "dot") or a SimilarityFunction object. Defaults to None.
            name (str, optional): The name of the evaluator. Defaults to "".
            show_progress_bar (bool, optional): Whether to show a progress bar during evaluation. Defaults to False.
            write_csv (bool, optional): Whether to write the evaluation results to a CSV file. Defaults to False.
            precision (Optional[Literal["float32", "int8", "uint8", "binary", "ubinary"]], optional): The precision
                to use for the embeddings. Can be "float32", "int8", "uint8", "binary", or "ubinary". Defaults to None.
            truncate_dim (Optional[int], optional): The dimension to truncate sentence embeddings to. `None` uses the
                model's current truncation dimension. Defaults to None.

        """
        super().__init__()
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.write_csv = write_csv
        self.precision = precision
        self.truncate_dim = truncate_dim

        assert len(self.sentences1) == len(self.sentences2)

        self.main_similarity = SimilarityFunction(main_similarity) if main_similarity else None
        self.name = name

        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs: Any) -> Self:
        """Create an instance of the class from a list of InputExample objects.

        Args:
            examples (list[InputExample]): A list of InputExample objects containing the input data.
            **kwargs (Any): Additional keyword arguments to be passed to the class constructor.

        Returns:
            Self: An instance of the class initialized with the provided input examples.

        """
        sentences1 = []
        sentences2 = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
        return cls(sentences1, sentences2, **kwargs)

    @classmethod
    def from_hf_dataset(cls, dataset: Dataset, **kwargs: Any) -> Self:
        """Create an EmbeddingDistanceEvaluator instance from a Hugging Face dataset.

        Args:
            dataset (Dataset): A Hugging Face dataset containing sentence pairs.
            **kwargs (Any): Additional keyword arguments to pass to the class constructor.

        Returns:
            EmbeddingDistanceEvaluator: An instance of the EmbeddingDistanceEvaluator class.

        """
        from datasets import Dataset  # noqa: F401

        sentences1 = []
        sentences2 = []

        for example in dataset:
            sentences1.append(example["sentence1"])
            sentences2.append(example["sentence2"])
        return cls(sentences1, sentences2, **kwargs)

    @override
    def __call__(
        self, model: SentenceTransformer, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float | int]:
        """Evaluate the model on a specified dataset.

           Calculates the mean and median pairwise distances between embeddings.

        Args:
            model (SentenceTransformer): The sentence transformer model to evaluate.
            output_path (str | None, optional): The path to save the evaluation results. Defaults to None.
            epoch (int, optional): The current epoch number. Defaults to -1.
            steps (int, optional): The number of steps in the current epoch. Defaults to -1.

        Returns:
            float: The mean pairwise distance between embeddings.

        Logs:
            Information about the evaluation process and the calculated distances.

        """
        if epoch != -1:
            out_txt = f" after epoch {epoch}" if steps == -1 else f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""

        logging.info("EmbeddingSimilarityEvaluator: Evaluating the model on the %s dataset %s:", self.name, out_txt)

        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
            embeddings1 = model.encode(
                self.sentences1,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=True,
                normalize_embeddings=False,
            )
            embeddings1 = embeddings1.to(torch.float64)
            embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1, eps=0)
            embeddings2 = model.encode(
                self.sentences2,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=True,
                normalize_embeddings=False,
            )
            embeddings2 = embeddings2.to(torch.float64)
            embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1, eps=0)

        similarity = torch.mm(embeddings1, embeddings2.T)
        ## calculating distances
        distances = 1 - similarity
        idx = torch.triu_indices(distances.size(0), distances.size(0), offset=1)
        upper_elements = distances[idx[0], idx[1]]

        zero_threshold = 1e-4
        median_distance = upper_elements.median()
        min_distance = upper_elements.min()
        min_distance_wo_zeros = upper_elements[upper_elements > zero_threshold].min()
        quantile_distance = torch.quantile(upper_elements, q=0.1)
        perc_of_zeros = torch.sum(upper_elements < zero_threshold) / len(upper_elements)

        self.primary_metric = "quantile_distance"
        metrics = {
            "median_distance": float(median_distance),
            "min_distance": float(min_distance),
            "min_distance_wo_zeros": float(min_distance_wo_zeros),
            "quantile_distance": float(quantile_distance),
            "perc_of_zeros": float(perc_of_zeros),
        }
        metrics = {self.name + "_" + key: value for key, value in metrics.items()}
        if hasattr(self, "primary_metric") and not self.primary_metric.startswith(self.name + "_"):
            self.primary_metric = self.name + "_" + self.primary_metric

        return metrics

    @override
    @property
    def description(self) -> str:
        return "Pairwise distances between Embeddings"
