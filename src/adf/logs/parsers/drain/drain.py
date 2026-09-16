# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import drain3.masking
import logging
import numpy as np
import polars as pl
import shutil
import tempfile as tf

from collections.abc import Iterable, Mapping
from drain3 import TemplateMiner
from drain3.drain import LogCluster
from drain3.file_persistence import FilePersistence
from drain3.masking import LogMasker
from functools import lru_cache
from numpy.typing import NBitBase, NDArray
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar
from typing_extensions import Self, override

from adf.core.common.logger import timelog as tl
from adf.core.dataframe.base import save_dataframe

from .base import BaseParser
from .config import DrainConfig
from .utils import _as_polars_dataframe, _check_clustering_metric, _check_X, _extract_masking

if TYPE_CHECKING:
    from torch import Tensor

    from adf.logs.models.sentence_embedder.sentence_embedder import SimCSETransformer

T = TypeVar("T", bound=NBitBase)

MEMORY_CACHE_SIZE = 2048


class DrainLogParser(BaseParser):
    """Class to parse logs for ADF."""

    str_replacement: str = "$1<<*>>"

    def __init__(
        self,
        template_miner: TemplateMiner | None = None,
        use_polars_masking: bool = False,
        trim_max_length: int | None = None,
    ) -> None:
        """Initialize the parser with the given parameters.

        Args:
            template_miner (TemplateMiner, optional): The template miner object. Defaults to TemplateMiner().
            use_polars_masking (bool, optional): Flag to use Rust regex for masking. Defaults to False.
            trim_max_length (int, optional): Maximum length to trim the log messages. Defaults to None.

        """
        self.template_miner = template_miner or TemplateMiner()
        self.rust_pattern = None
        self.trim_max_length = trim_max_length

        if use_polars_masking:
            self.rust_pattern = _extract_masking(template_miner=self.template_miner)
            self.template_miner.masker = LogMasker([], mask_prefix=None, mask_suffix=None)

    @classmethod
    def load_from_dirpath(
        cls, dirpath: str | Path, preserve_init_state: bool = True, tmp_subdir: str | None = None, **kwargs: Any
    ) -> Self:
        """Create an instance of the class from the given state and configuration file paths.

        Args:
            dirpath (str | Path): The directory path containing the state and configuration files.
            preserve_init_state (bool, optional): Flag to preserve the original state file. Defaults to True.
            tmp_subdir (str, optional): The temporary directory to use. Defaults to None.
            **kwargs (Any): Additional keyword arguments to pass to the class constructor.

        Returns:
            Self: An instance of the class.

        Raises:
            FileNotFoundError: If the configuration file or state file is not found.

        """
        if not Path(dirpath).exists() or not Path(dirpath).is_dir():
            raise NotADirectoryError(f"Directory {dirpath} does not exist or is not a directory.")

        dirsp = list(Path(dirpath).glob("[!.]*.bin"))  # Exclude metadata files created by tar

        if not dirsp:
            file_persistence = FilePersistence(Path(tf.mkdtemp(dir=tmp_subdir)).joinpath("state.bin"))
        else:
            if not dirsp[0].is_file():
                raise FileNotFoundError(f"State file not found: {dirsp[0]}")
            elif dirsp[0].is_file() and preserve_init_state:
                tmp_file = Path(tf.mkdtemp(dir=tmp_subdir)).joinpath("state.bin")
                shutil.copyfile(dirsp[0], tmp_file)
                file_persistence = FilePersistence(tmp_file)
            else:
                file_persistence = FilePersistence(dirsp[0])

        dircfg = list(Path(dirpath).glob("[!.]*.ini"))  # Exclude metadata files created by tar
        if not dircfg:
            logging.warning("Configuration file not found. Setting default configuration.")
            cfg = DrainConfig()
        else:
            cfg = DrainConfig().load(dircfg[0])

        template_miner = TemplateMiner(config=cfg, persistence_handler=file_persistence)
        return cls(template_miner=template_miner, **kwargs)

    @lru_cache(maxsize=MEMORY_CACHE_SIZE)
    def match(self, log: str) -> int:
        """Match the log message to a template.

        Args:
            log (str): The log message to match.

        Returns:
            str: The matched template.

        """
        # Match the log message to a cluster template. If template is not matched, return -1.

        cid = self.template_miner.match(log)
        return cid.cluster_id if cid else -1

    def predict(self, X: Iterable[str]) -> list[int]:
        """Match the log messages in the array to templates.

        Args:
            X (Collection[str]): The list of log

        Returns:
            list: A list of matched templates.

        """
        logs = [self.match(log) for log in X]
        return logs

    def discard_cluster(self, cid: int, clear_cache: bool = False) -> None:
        """Removes a cluster from the drain template miner in order to not be matched.

        Args:
            cid (int): The cluster ID
            clear_cache (bool, optional): Whether to clear the cache for the match function. Defaults to False

        """
        if cid in self.template_miner.drain.id_to_cluster:
            del self.template_miner.drain.id_to_cluster[cid]
        if clear_cache:
            self.match.cache_clear()

    @staticmethod
    @tl
    def transform_(
        sentences: Path | list[Path] | pl.DataFrame | pl.LazyFrame | list[str] | NDArray[np.str_],
        pattern: list[tuple[str, list[drain3.masking.MaskingInstruction]]] | None = None,
        return_unique: bool = True,
        slice_length: int | None = None,
        rstrip: bool = True,
        **kwargs: dict[str, Any],
    ) -> list[str]:
        """Preprocess the given sentences.

        Args:
            sentences (Iterable[str]): The sentences to preprocess.
            pattern (str, optional): The regex pattern to apply to the sentences. Defaults to None.
            return_unique (bool, optional): Whether to return the unique sentences. Defaults to True.
            slice_length (int, optional): The trim length of the sentences. Defaults to None.
            rstrip (bool, optional): Whether to remove the leading and trailing whitespace. Defaults to True.
            **kwargs (Any): Additional keyword arguments to pass to the constructor of the polars dataframe.

        """
        return _check_X(
            _as_polars_dataframe(sentences, **kwargs),
            return_unique=return_unique,
            pattern=pattern,
            slice_length=slice_length,
            rstrip=rstrip,
        )

    @tl
    def parse(
        self, sentences: Path | list[Path] | pl.DataFrame | list[str] | NDArray[np.str_], **kwargs: Any
    ) -> list[int]:
        """Parse the input data and processes it according to the specified parameters.

        Args:
            sentences (str | Path | pl.DataFrame): The input data, which can be a file path or a Polars DataFrame.
            **kwargs (Any): Additional keyword arguments to pass to the function.

        Returns:
            pl.DataFrame | tuple[pl.DataFrame, list[int]]: The processed DataFrame, and optionally the matched array
                if return_matched_array is True.

        """
        X = self.transform_(
            sentences, return_unique=False, pattern=self.rust_pattern, slice_length=self.trim_max_length, **kwargs
        )

        preds = self.predict(X)
        return preds

    @tl
    @override
    def fit(self, sentences: Path | list[Path] | pl.DataFrame | list[str] | NDArray[np.str_], **kwargs: Any) -> None:
        """Fits the log parser model to the provided dataframe.

        Args:
            sentences (str | Path | pl.DataFrame): The input data, which can be a file path or a Polars DataFrame.
            **kwargs (Any): Additional keyword arguments to pass to the function.

        """
        X = self.transform_(
            sentences, return_unique=True, pattern=self.rust_pattern, slice_length=self.trim_max_length, **kwargs
        )

        # Disable FilePersistence to avoid writing to disk while fitting
        persistence_handler = self.template_miner.persistence_handler
        self.template_miner.persistence_handler = None

        for log in X:
            self.template_miner.add_log_message(log)

        # Re-enable FilePersistence
        self.template_miner.persistence_handler = persistence_handler

        if self.template_miner.persistence_handler is not None:
            self.template_miner.save_state("full")

    @override
    def score(
        self,
        sentences: list[str],
        *,
        sentence_transformer: str | SimCSETransformer = "all-MiniLM-L6-v2",
        sentence_transformer_kwargs: Mapping[str, Any] | None = None,
        encode_kwargs: Mapping[str, Any] | None = None,
        metric: str | Iterable[str] = "calinski_harabasz_score",
    ) -> np.number[T] | list[np.number[T]]:
        """Return the score of the template miner.

        Args:
            sentences (list[str]): The list of sentences to score.
            sentence_transformer (str | SimCSETransformer): The sentence transformer model to use.
            sentence_transformer_kwargs (Mapping[str, Any], optional): Additional keyword arguments
                to pass to the sentence transformer. Defaults to None.
            encode_kwargs (Mapping[str, Any], optional): Additional keyword arguments to pass to the encode function.
                Defaults to None.
            metric (str | Iterable[str]): The metric to use for scoring. Defaults to "calinski_harabasz_score".

        Args:
            sentences (list[str]): The list of sentences to score.
            sentence_transformer (str | SimCSETransformer): The sentence transformer model to use.
            sentence_transformer_kwargs (Mapping[str, Any], optional): Additional keyword arguments
                to pass to the sentence transformer. Defaults to None.
            encode_kwargs (Mapping[str, Any], optional): Additional keyword arguments to pass to the encode function.
                Defaults to None.
            metric (str | Iterable[str]): The metric to use for scoring. Defaults to "calinski_harabasz_score".

        Returns:
            np.number[T] | list[np.number[T]]: The score(s) of the template miner.

        """
        if sentence_transformer_kwargs is None:
            sentence_transformer_kwargs = dict()

        if encode_kwargs is None:
            encode_kwargs = dict()

        x_embed = self._embed(
            sentences,
            sentence_transformer=sentence_transformer,
            sentence_transformer_kwargs=sentence_transformer_kwargs,
            encode_kwargs=encode_kwargs,
        )
        labels = self.predict(sentences)

        if isinstance(metric, str):
            return _check_clustering_metric(metric)(x_embed, labels)
        elif isinstance(metric, list):
            return [_check_clustering_metric(scorer)(x_embed, labels) for scorer in metric]
        else:
            raise ValueError(f"Invalid metric type: {type(metric)}")

    def _embed(
        self,
        sentences: str | list[str],
        *,
        sentence_transformer: str | SimCSETransformer,
        sentence_transformer_kwargs: Mapping[str, Any] | None = None,
        encode_kwargs: Mapping[str, Any] | None = None,
    ) -> NDArray[np.str_] | Tensor | list[Tensor]:
        """Return the embeddings of the templates.

        Args:
            sentences (str | list[str]): The list of sentences to embed.
            sentence_transformer (str): The sentence transformer model to use.
            sentence_transformer_kwargs (Mapping[str, Any], optional): Additional keyword arguments
                to pass to the sentence transformer. Defaults to None.
            encode_kwargs (Mapping[str, Any], optional): Additional keyword arguments to pass to the encode function.
                Defaults to None.

        Returns:
            NDArray[np.str_] | Tensor | list[Tensor]: The embeddings of the templates.

        """
        sentence_transformer_kwargs = sentence_transformer_kwargs or dict()
        encode_kwargs = encode_kwargs or {"batch_size": 128, "show_progress_bar": True}
        from adf.logs.models.sentence_embedder.sentence_embedder import SimCSETransformer as _SimCSETransformer

        st = (
            _SimCSETransformer(sentence_transformer, **sentence_transformer_kwargs)
            if isinstance(sentence_transformer, str)
            else sentence_transformer
        )

        if isinstance(sentences, str):
            sentences = " ".join(self.template_miner.drain.get_content_as_tokens(sentences))
        else:
            sentences = [" ".join(self.template_miner.drain.get_content_as_tokens(sentence)) for sentence in sentences]

        return st.encode(sentences, **encode_kwargs)  # type: ignore[no-any-return]

    @property
    def config(self) -> DrainConfig | None:
        """Returns the path to the configuration file.

        Returns:
            cfg (str): The path to the configuration file.

        """
        cfg = getattr(self.template_miner, "config", None)
        return cfg

    @property
    def state_path(self) -> Path | None:
        """Returns the path to the state file.

        Returns:
            (Path | None): The path to the state file.

        """
        ph = getattr(self.template_miner.persistence_handler, "file_path", None)
        return Path(ph) if ph else None

    @property
    @override
    def templates(self) -> list[LogCluster]:
        """Property that returns the clusters of templates from the template miner.

        Returns:
            templates (list): A list of LogCluster objects.

        """
        return list(self.template_miner.drain.clusters)

    @property
    def cluster_ids(self) -> list[int]:
        """Returns the cluster ids of the templates.

        Returns:
            cluster_ids (list): A list of cluster ids.

        """
        return [template.cluster_id for template in self.templates]

    def save_templates(self, path: str | Path = "./") -> None:
        """Store the templates extracted by the template miner into a DataFrame.

        Args:
            path (str): The path where the templates will be stored.

        """
        templates = self.templates

        save_dataframe(
            data={
                "ClusterId": [template.cluster_id for template in templates],
                "Template": [" ".join(template.log_template_tokens) for template in templates],
                "Occurrences": [template.size for template in templates],
            },
            destination=Path(path),
        )

    @override
    def save_model(self, path: str | Path = "./") -> None:
        """Save the current state of the parser to a specified path.

        Args:
            path (str): The local file path where the state will be saved.

        """
        # Save the state
        if self.state_path is not None:
            self.template_miner.save_state("full")
            shutil.copyfile(self.state_path, Path(path).joinpath("state.bin"))
        else:
            fp = FilePersistence(Path(path).joinpath("state.bin"))
            self.template_miner.persistence_handler = fp
            self.template_miner.save_state("full")

        # Save the configuration
        if self.config is not None:
            self.config.save(Path(path).joinpath("config.ini"))
        else:
            cfg = DrainConfig()
            cfg.save(Path(path).joinpath("config.ini"))
