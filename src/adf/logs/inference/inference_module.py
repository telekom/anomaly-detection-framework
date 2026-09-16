# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import torch

from contextlib import nullcontext
from sentence_transformers.util import get_device_name
from torch import Tensor
from torch.nn.functional import pad
from typing import Any, Literal, TypeGuard, cast

# from typing_extensions import overload
from adf.core.common.logger import timelog as tl
from adf.logs.inference.registry import add_to_logs_registry
from adf.logs.models.autoencoder.model import LogAutoEncoder
from adf.logs.models.sentence_embedder.sentence_embedder import SimCSETransformer
from adf.logs.parsers.drain.drain import DrainLogParser


def is_nested_list(x: list[str] | list[list[str]]) -> TypeGuard[list[list[str]]]:
    """Type guard to check if a list is a nested list of strings.

    Args:
        x (list[str] | list[list[str]]): The list to check.

    Returns:
        TypeGuard[list[list[str]]]: True if the list is a nested list of strings, False otherwise.

    """
    return isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)


@add_to_logs_registry
class DSAInferencePipeline:
    def __init__(
        self,
        parser_model_or_path: str | DrainLogParser,
        sentence_transformer_model_or_path: str | SimCSETransformer,
        autoencoder_model_or_path: str | LogAutoEncoder,
        *,
        pre_calculated_cache: str | dict[str, Any] | None = None,
        threshold: str | float | None = None,
        window_size: int = 40,
        embed_size: int = 384,
        device_sentence_transformer: str | None = None,
        device_autoencoder: str | None = None,
        parsing_kwargs: dict[str, Any] | None = None,
        torch_compile: Literal["transformer", "autoencoder", "both", "none"] = "none",
        torch_compile_kwargs: dict[str, dict[str, Any]] | None = None,
        encode_kwargs: dict[str, Any] | None = None,
        torch_mm32_precision: Literal["highest", "high", "medium"] = "high",
        use_autocast: bool = False,
    ):
        """Initialize the InferenceModule with the given parameters.

        Args:
            parser_model_or_path (DrainLogParser): The log parser to be used.
            sentence_transformer_model_or_path (SimCSETransformer): The sentence transformer model.
            autoencoder_model_or_path (LogAutoEncoder): The autoencoder model.
            pre_calculated_cache (str | dict[str, Any] | None): Local path or dict of pre-calculated embeddings.
            threshold (float | None, optional): The threshold for anomaly detection. Defaults to None.
            window_size (int, optional): The size of the window for processing. Defaults to 40.
            embed_size (int, optional): The size of the embeddings. Defaults to 384.
            device_sentence_transformer (str | None, optional): The device for the sentence transformer model.
                Defaults to None.
            device_autoencoder (str | None, optional): The device for the autoencoder model. Defaults to None.
            encode_kwargs (dict[str, Any], optional): Additional keyword arguments for the encode method
                of the sentence_transformer. Defaults to None.
            parsing_kwargs (dict[str, Any], optional): Additional keyword arguments for initializing Drain parser.
            torch_compile (Literal["transformer", "autoencoder", "both"], optional): The model to compile.
            torch_compile_kwargs (dict[str, dict[str, Any]], optional): Additional keyword arguments for torch.compile.
            torch_mm32_precision (Literal["highest", "high", "medium"], optional): The precision for torch operations.
            use_autocast (bool, optional): If True, use autocast for mixed precision. Defaults to False.

        """
        self.encode_kwargs = encode_kwargs or {}
        self.torch_compile_kwargs = torch_compile_kwargs or {}
        self.parsing_kwargs = parsing_kwargs or {}

        self.window_size = window_size
        self.embed_size = embed_size

        self.threshold = (
            self.set_threshold_from_filepath(filepath=autoencoder_model_or_path, threshold=threshold)
            if isinstance(autoencoder_model_or_path, str) and isinstance(threshold, str)
            else (
                logging.warning(
                    "To load the threshold from a .ckpt file, both 'autoencoder_model_or_path' and 'threshold'"
                    "must be file paths (str). setting to None."
                )  # type:ignore[func-returns-value]
                or None
                if isinstance(threshold, str)
                else threshold
            )
        )

        self.device_sentence_transformer = get_device_name() or device_sentence_transformer
        self.device_autoencoder = get_device_name() or device_autoencoder

        self.parser = (
            DrainLogParser.load_from_dirpath(parser_model_or_path, **self.parsing_kwargs)
            if isinstance(parser_model_or_path, str)
            else parser_model_or_path
        )

        self.autoencoder = (
            LogAutoEncoder.load_from_checkpoint(checkpoint_path=autoencoder_model_or_path)
            if isinstance(autoencoder_model_or_path, str)
            else autoencoder_model_or_path
        )
        self.st = (
            SimCSETransformer(sentence_transformer_model_or_path)
            if isinstance(sentence_transformer_model_or_path, str)
            else sentence_transformer_model_or_path
        )

        # Specify model dtype and device
        self.tensor_dtype = self.autoencoder.model_dtype
        self.autoencoder.to(self.device_autoencoder)
        self.st.to(self.device_sentence_transformer)

        # Compile the models after moving them to the correct device
        self.torch_compile = torch_compile
        self.compile()

        # Set the models to evaluation mode
        self.autoencoder.eval()
        self.st.eval()

        # Create a cache for storing embeddings or load it if it's pre-calculated
        self.load_pre_calculated_cache(
            pre_calculated_cache
        ) if pre_calculated_cache is not None else self.create_cache()

        # Set the precision for torch operations
        torch.set_float32_matmul_precision(torch_mm32_precision)
        self.amp_context = (
            torch.amp.autocast(enabled=True, device_type=self.device_autoencoder, cache_enabled=True)  # type: ignore
            if use_autocast
            else nullcontext()
        )

    @staticmethod
    def set_threshold_from_filepath(filepath: str, threshold: str | float | None = None) -> Any:
        """Load threshold from the filepath.

        Args:
            filepath (str): The filepath to load the threshold from.
            threshold (str | float | None): The threshold to load.

        """
        if isinstance(threshold, str):
            return torch.load(filepath, weights_only=False).get(threshold, None)
        else:
            raise ValueError(
                "To load the threshold from a .ckpt file, both 'autoencoder_model_or_path' and 'threshold' must be str."
                "To set the 'threshold' manually, the 'threshold' must be a float instance."
            )

    def load_pre_calculated_cache(self, pre_calculated_cache: str | dict[str, Any]) -> None:
        """Load the embedding cache without building it.

        Args:
            pre_calculated_cache (str | dict[str, Any]): The cache to load the embedding from.

        Sets:
            cluster_id_to_tensor (dict[str, Tensor]): The embeddings cache.

        """
        if isinstance(pre_calculated_cache, str):
            self.cluster_id_to_tensor = torch.load(pre_calculated_cache, map_location=self.device_sentence_transformer)

        elif isinstance(pre_calculated_cache, dict):
            self.cluster_id_to_tensor = pre_calculated_cache
        else:
            raise ValueError("pre_calculated_cache must be str or dict")

        # assert device is the same with sentence_transformer
        if list(self.cluster_id_to_tensor.values())[0].device.type != self.device_sentence_transformer:
            raise RuntimeError(
                f"cluster_id_to_tensor values must be on the same device as sentence_transformer."
                f"Expected {self.device_sentence_transformer} got {list(self.cluster_id_to_tensor.values())[0].device}"
            )
        if len(self.cluster_id_to_tensor.keys()) != len(self.parser.templates):
            raise RuntimeError(
                f"Expected same number of keys in cluster_id_to_tensor with parser templates. "
                f"Expected {len(self.parser.templates)} got {len(self.cluster_id_to_tensor.keys())}"
            )

    def create_cache(self) -> None:
        """Create a cache for storing embeddings and their corresponding log templates.

        This method initializes two dictionaries:
        - `cluster_id_to_tensor`: Maps cluster IDs to their corresponding tensor embeddings.

        The embeddings are generated using the sentence transformer model (`self.st.encode`).
        If a log template is already cached, its precomputed embedding is used.

        Returns:
            None

        """
        parser_templates = self.parser.templates

        # Create a cache for storing the embeddings and the templates
        logging.info("Loaded %d log templates. Creating embedding cache.", len(parser_templates))
        self.cluster_id_to_tensor = {}

        tensors = self.st.encode(
            [" ".join(template.log_template_tokens) for template in parser_templates],
            convert_to_tensor=True,
            **self.encode_kwargs,
        )

        for i, t in enumerate(parser_templates):
            self.cluster_id_to_tensor[t.cluster_id] = tensors[i].unsqueeze(0)

    def sync_cache(self, template_ids: list[str | int], tensors: Tensor) -> None:
        """Add template embeddings to the cache by their cluster IDs.

        This method updates the `cluster_id_to_tensor` cache by associating template IDs
        with their tensor embeddings. No Drain parsing is performed - templates are
        pre-parsed and IDs are provided directly.

        Args:
            template_ids (list[str]): A list of cluster IDs corresponding to the templates.
            tensors (Tensor): A tensor containing the embeddings corresponding to the templates.

        Sets:
            - `self.cluster_id_to_tensor`: A dictionary mapping cluster IDs to their tensor representations.

        """
        logging.info(f"Adding {len(template_ids)} template embeddings to cache.")
        for i, template_id in enumerate(template_ids):
            self.cluster_id_to_tensor[template_id] = tensors[i].unsqueeze(0)

    @tl
    def dynamic_encode_from_templates(
        self, templates: list[str | list[str]], template_ids: list[str | int | list[str] | list[int]]
    ) -> Tensor:
        """Dynamically encode a batch of template sequences into tensors.

        This method processes a batch of template sequences with their corresponding IDs,
        where the first sequence is used to initialize the encoding process. Subsequent
        sequences are encoded using a sliding window approach. The resulting tensors are
        concatenated to form the final batch.

        Args:
            templates (list[str | list[str]]): First element is the full window ``list[str]``;
                remaining elements are scalar template strings (same shape as serving JSON ``temp_data``).
            template_ids (list[str | int | list[str] | list[int]]): Parallel to ``templates``;
                first element is the window of IDs; remaining entries are scalars (JSON may use int).

        Returns:
            Tensor: A tensor containing the encoded template sequences. The shape of the tensor
                depends on the input template sequences and the window size.

        """
        if not templates or not template_ids:
            raise ValueError("templates and template_ids cannot be empty")
        if len(templates) != len(template_ids):
            raise ValueError("templates and template_ids must have the same length")

        # If the dynamic batch contains only the first full window, encode it as a single window batch.
        if len(templates) == 1:
            first = templates[0]
            first_ids = template_ids[0]
            if not isinstance(first, list) or not isinstance(first_ids, list):
                raise TypeError("Expected first element of dynamic batch to be a list[str] (full window).")
            return self.encode_from_templates(first, first_ids).unsqueeze(0)  # shape: (1, window_size, embed_size)

        # Encode first window (full window of templates)
        tw, tid0 = templates[0], template_ids[0]
        if not isinstance(tw, list) or not isinstance(tid0, list):
            raise TypeError("Expected first element of dynamic batch to be list (full window).")
        _first = self.encode_from_templates(tw, tid0)  # (window_size, embed_size)

        # Remaining entries are scalar templates and IDs
        # (e.g. dynamic batch payload: [["t"]*40, "t"], ids: [[1]*40, 1]).
        _rest = self.encode_from_templates(
            cast(list[str], templates[1:]), cast(list[str | int], template_ids[1:])
        )  # (N-1, embed_size)

        return torch.cat([_first, _rest], dim=0).unfold(0, self.window_size, 1).transpose(-2, -1)

    @tl
    def encode_from_templates(
        self,
        templates: str | list[str],
        template_ids: str | int | list[str] | list[int] | list[str | int],
        *,
        sync_cache: bool = True,
    ) -> Tensor:
        """Encode templates into tensors using their cluster IDs.

        This method encodes templates by looking up their embeddings in the cache using
        the provided template IDs. If an ID is not found in the cache, the corresponding
        template is encoded using the sentence transformer and added to the cache.

        Args:
            templates (str | list[str]): A single template or list of templates to be processed.
            template_ids (str | int | list[str] | list[int] | list[str | int]): Cluster ID(s); lists
                may be all-str, all-int, or mixed (e.g. serving JSON ``ids`` tail).
            sync_cache (bool): If True, add new template embeddings to the cache. Defaults to True.

        Returns:
            Tensor: The encoded tensor representation of the templates.

        """
        # Normalize inputs to lists
        template_list = templates if isinstance(templates, list) else [templates]
        if isinstance(template_ids, list):
            id_list: list[str | int] = cast(list[str | int], template_ids)
        else:
            id_list = [template_ids]

        if len(template_list) != len(id_list):
            raise ValueError("templates and template_ids must have the same length")

        # Try to get cached embeddings
        cached_tensors = []
        missed_indices = []
        missed_templates = []
        missed_ids: list[str | int] = []

        for i, (template, template_id) in enumerate(zip(template_list, id_list)):
            if template_id in self.cluster_id_to_tensor:
                cached_tensors.append((i, self.cluster_id_to_tensor[template_id]))
            else:
                missed_indices.append(i)
                missed_templates.append(template)
                missed_ids.append(template_id)

        # If all templates are cached, return concatenated cached tensors
        if not missed_indices:
            return torch.cat([tensor for _, tensor in cached_tensors], dim=0)

        # Encode missed templates
        missed_embeddings = self.st.encode(
            missed_templates, convert_to_tensor=True, show_progress_bar=False, **self.encode_kwargs
        )

        # Update cache with new template embeddings
        if sync_cache:
            self.sync_cache(missed_ids, missed_embeddings)

        # Build final embedding tensor
        emb = torch.empty((len(template_list), self.embed_size), device=self.device_sentence_transformer)

        # Fill cached positions
        for idx, tensor in cached_tensors:
            emb[idx] = tensor.squeeze(0)

        # Fill missed positions
        for i, missed_idx in enumerate(missed_indices):
            emb[missed_idx] = missed_embeddings[i]

        return emb

    @tl
    def infer(
        self,
        templates: list[str] | list[list[str]],
        template_ids: list[str] | list[list[str]],
        *,
        return_class: bool = False,
        dynamic_batching: bool = True,
        force_pad_to_sequence_length: bool = False,
    ) -> Tensor:
        """Perform inference on a sequence of templates.

        This method processes a sequence of templates with their corresponding IDs, encoding
        each template into a tensor. If a template ID exists in the cache, the precomputed
        tensor is used. Otherwise, the template is encoded using the sentence transformer and
        added to the cache. The resulting tensors are then passed to the autoencoder for inference.

        Args:
            templates (list[str] | list[list[str]]): A list of templates to be processed or a batch of them.
                For dynamic batching, first element should be a list (full window), rest are individual templates.
            template_ids (list[str] | list[list[str]]): A list of template IDs corresponding to the templates.
                Must match the structure of templates.
            dynamic_batching (bool): If True, the batch is constructed dynamically. The first template list is used to
                determine the batch size. The rest of the templates are constructed in a sliding window manner based
                on the first template list. Defaults to True.
            return_class (bool): If True and threshold is given, return the class of the template.
                1 for anomalous and 0 for normal. Defaults to False.
            force_pad_to_sequence_length (bool, optional): Whether to force the padding to the sequence length
                (Add timesteps to window that have zero embeddings). Defaults to False.

        Returns:
            Tensor: The output tensor from the autoencoder.

        """
        if not isinstance(templates, list) or not isinstance(template_ids, list):
            raise TypeError("templates and template_ids must be lists.")

        if len(templates) != len(template_ids):
            raise ValueError("templates and template_ids must have the same length.")

        if dynamic_batching and force_pad_to_sequence_length:
            raise ValueError("Dynamic batching cannot be applied with force_pad_to_sequence_length.")

        if dynamic_batching:
            # For dynamic batching, templates and template_ids should be in format:
            # templates = [["template1", "template2", ..., "template40"], "template41", "template42", ...]
            # template_ids = [["1", "156", ..., "589"], "1601", "22", ...]
            if not isinstance(templates[0], list) or not isinstance(template_ids[0], list):
                raise TypeError(
                    "For dynamic batching, first element of templates and template_ids must be lists (full window)."
                )
            inp = self.dynamic_encode_from_templates(templates, template_ids)  # type: ignore
        else:
            # For non-dynamic batching, templates and template_ids should be flat lists
            if isinstance(templates[0], list) or isinstance(template_ids[0], list):
                raise TypeError("For non-dynamic batching, templates and template_ids must be flat lists.")
            inp = self.encode_from_templates(templates, template_ids).unsqueeze(0)  # type: ignore

        mask = None
        if force_pad_to_sequence_length and inp.size(1) < self.window_size:
            padding = (0, 0, 0, self.window_size - inp.size(1), 0, 0)  # add zero embeddings to timesteps of the window
            inp = pad(inp, padding)
            mask = inp.ne(0).any(dim=-1, keepdim=True).expand_as(inp)

        if inp.size(1) != self.window_size or inp.size(2) != self.embed_size:
            raise ValueError(
                f"Expected input shape (batch_size, {self.window_size}, {self.embed_size}), but got {inp.size()}"
            )

        inp = inp.to(self.device_autoencoder, dtype=self.tensor_dtype)

        with self.amp_context:
            reconstruction_error = self.autoencoder.predict_single(inp, mask=mask, aggregate="all")

        if return_class:
            if not self.threshold:
                raise ValueError("Threshold not set. Set threshold when inferring with return_class=True.")
            return (reconstruction_error > self.threshold).to(torch.int16)

        return reconstruction_error

    def compile(self) -> None:
        """Compile the models for faster inference.

        This method compiles the sentence transformer and autoencoder models using
        torch.compile for optimized performance.

        """
        match self.torch_compile:
            case "transformer":
                self.st.compile(**self.torch_compile_kwargs.get("transformer", {}))
            case "autoencoder":
                self.autoencoder.compile(**self.torch_compile_kwargs.get("autoencoder", {}))
            case "both":
                self.st.compile(**self.torch_compile_kwargs.get("transformer", {}))
                self.autoencoder.compile(**self.torch_compile_kwargs.get("autoencoder", {}))
            case "none":
                pass
            case _:
                raise ValueError(
                    f"Invalid torch_compile value: {self.torch_compile}. "
                    "Expected 'transformer', 'autoencoder', or 'both'."
                )
