# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect

from collections.abc import Callable
from omegaconf import OmegaConf
from pathlib import Path
from typing import Any, TypeVar
from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T", bound=Callable[..., Any])

LOGS_MODEL_REGISTRY = dict[str, Any]()


def add_to_logs_registry(cls: Any) -> Any:
    """Register a class decorator and its constructor arguments in the `REGISTER` dictionary.

    This decorator ensures that the class and its required arguments are stored in the `REGISTER`
    dictionary for later use. The `REGISTER` dictionary maps the class name to a list of its
    constructor arguments, excluding `self`.

    Args:
        cls (Any): The class to be registered.

    Returns:
        Any: The original class, unmodified.

    Raises:
        ValueError: If the decorator is applied to a non-class object.

    Examples:
        >>> from adf.logs.inference.registry import add_to_logs_registry
        >>> @add_to_logs_registry
        >>> class AwesomeClass:
        ...     def __init__(self, arg1, arg2, *args, **kwargs):
        ...         pass
        >>> print(LOGS_MODEL_REGISTRY)
        {'AwesomeClass': ['arg1', 'arg2', *args]}

    """
    if not inspect.isclass(cls):
        raise ValueError("Decorator must be applied to a class.")

    required = inspect.getfullargspec(cls).args
    required.remove("self")

    optional = inspect.getfullargspec(cls).kwonlyargs

    LOGS_MODEL_REGISTRY[cls.__name__] = [required, optional]
    return cls


def model_config_loader(config_path: str | Path) -> Any:
    """Load models from a configuration file helper function.

    Args:
        config_path (str | Path): Path to the configuration file.

    Returns:
        Any: An instance of the model inference class with the loaded models.

    """
    importlib.import_module("adf.logs.inference.inference_module")

    try:
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    except Exception as e:
        raise ValueError(f"Failed to load config file: {config_path}") from e

    if not isinstance(config, dict):
        raise ValueError(f"Configuration file {config_path} is not a valid dictionary.")

    model_type = config.get("model_type", None)
    if model_type is None:
        raise KeyError("model_type must be specified in the configuration file.")

    if model_type not in LOGS_MODEL_REGISTRY or not isinstance(model_type, str):
        raise KeyError(
            f"Model type {model_type} is not registered or not isinstance `str`. "
            f"Available types: {list(LOGS_MODEL_REGISTRY.keys())}"
        )

    required_params, optional_params = LOGS_MODEL_REGISTRY[model_type]

    for param in required_params:
        if param not in config:
            raise KeyError(f"Missing required parameter '{param}' in the configuration file.")

    module = getattr(importlib.import_module("adf.logs.inference.inference_module"), model_type)
    kwds = {param: config[param] for param in required_params}
    kwds.update({param: config[param] for param in optional_params if param in config})

    return module(**kwds)
