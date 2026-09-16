# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import torch.nn as nn

from collections.abc import Callable
from typing import Any, TypeVar, cast
from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T", bound=Callable[..., Any])

MODEL_REGISTRY = dict[str, Any]()


def add_to_logs_model_registry(cls: Any) -> Any:
    """Register a class decorator and its constructor arguments in the `MODEL_REGISTRY` dictionary.

    This decorator ensures that the class and its required arguments are stored in the `MODEL_REGISTRY`
    dictionary for later use. The `MODEL_REGISTRY` dictionary maps the class name to a list of its
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
        >>> print(MODEL_REGISTRY)
        {'AwesomeClass': ['arg1', 'arg2', *args]}

    """
    if not inspect.isclass(cls):
        raise ValueError("Decorator must be applied to a class.")

    required = inspect.getfullargspec(cls).args
    required.remove("self")

    optional = inspect.getfullargspec(cls).kwonlyargs

    MODEL_REGISTRY[cls.__name__] = [required, optional]
    return cls


def model_loader(model_name: str, model_args: dict[str, Any] | None = None) -> nn.Module:
    """Load a model from the registry.

    Args:
        model_name (str): The name of the model to load.
        model_args (dict[str, Any] | None): The arguments to pass to the model.

    Returns:
        nn.Module: The loaded model.

    Note:
        When adding a model to the registry using :function:`add_to_logs_model_registry`, ensure the model is properly
        imported and initialized in the `__init__.py` file. This function will look at all imported statements in the
        adf.logs.models.autoencoder.__init__.py

    """
    model_args = model_args or {}

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} is not registered.")

    module = getattr(__import__("adf.logs.models.autoencoder", fromlist=[model_name]), model_name)
    return cast(nn.Module, module(**model_args))
