# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import sys
import torch
import types

from collections.abc import Iterable, Mapping
from lightning.pytorch.utilities.types import LRSchedulerConfig
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from typing import Any, cast

logger = logging.getLogger(__name__)


def clear_stale_mpi_env() -> list[str]:
    """Keep Lightning Trainer off OpenMPI on PyTorch containers that ship mpi4py.

    Real mpi4py auto-inits OpenMPI and aborts. Lightning 2.5.5 ``detect()`` does
    not catch ImportError, so a stub without ``MPI`` also crashes. Install a fake
    MPI with world size 1 and force ``MPIEnvironment.detect`` to False.
    """

    class _FakeComm:
        def Get_size(self) -> int:
            return 1

        def Get_rank(self) -> int:
            return 0

    fake_mpi = types.ModuleType("mpi4py.MPI")
    fake_mpi.COMM_WORLD = _FakeComm()  # type: ignore[attr-defined]
    fake_mpi.COMM_SELF = _FakeComm()  # type: ignore[attr-defined]

    pkg = types.ModuleType("mpi4py")
    pkg.__path__ = []
    pkg.MPI = fake_mpi  # type: ignore[attr-defined]
    sys.modules["mpi4py"] = pkg
    sys.modules["mpi4py.MPI"] = fake_mpi

    try:
        from lightning.fabric.plugins.environments.mpi import MPIEnvironment

        def _detect_disabled() -> bool:
            return False

        MPIEnvironment.detect = staticmethod(_detect_disabled)  # type: ignore[method-assign]
        logger.info("Forced Lightning MPIEnvironment.detect() to False")
    except ImportError:
        pass

    logger.info("Installed fake mpi4py (COMM_WORLD size=1) so Trainer skips MPI")

    removed: list[str] = []
    for key in list(os.environ):
        if key.startswith(("OMPI_", "PMIX_")) or key in {"PMI_RANK", "PMI_SIZE", "PMI_FD"}:
            del os.environ[key]
            removed.append(key)
    if removed:
        logger.info("Cleared stale MPI environment variables: %s", ", ".join(sorted(removed)))
    return removed


def reduce(
    input: Tensor, reduction: str = "mean", dim: int | list[int] | None = None, mask: Tensor | None = None
) -> Tensor:
    """Reduce a tensor along the specified dimension.

    Args:
        input (Tensor): input tensor to be reduced along the specified dimension.
        reduction (Literal["mean", "sum", "none"], optional): Reduction to apply to the reduced tensor.
            Defaults to "mean".
        dim (int | list[int], optional): Dimension along which to reduce. Defaults to 0.
        mask (Tensor | None): Mask to apply to the reduced tensor. Defaults to None.

    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"Reduction {reduction} not supported. Choose from {['mean', 'sum', 'none']}")

    if mask is None:
        mask = torch.ones_like(input, dtype=input.dtype, device=input.device)

    if reduction == "mean":
        return torch.sum(input * mask, dim=dim) / torch.sum(mask, dim=dim) + 1e-8  # stability factor
    elif reduction == "sum":
        return torch.sum(input * mask, dim=dim)
    else:
        return input * mask


def construct_optimizer(
    params: Iterable[Tensor] | Iterable[Mapping[str, Any]],
    optim_name: str,
    optim_config: Mapping[str, Any] | None = None,
) -> Optimizer:
    """Build and return a PyTorch optimizer.

    Args:
        params (Iterable[Tensor] | Iterable[dict[str, Any]]):
            Parameters to be optimized by the optimizer.
        optim_name (str):
            Name of the optimizer to be used.
        optim_config (dict[str, Any], optional):
            Additional keyword arguments to be passed to the optimizer. Defaults to an empty dictionary.

    Returns:
        Optimizer:
            An instance of the specified optimizer.

    Raises:
        ValueError:
            If the specified optimizer name is not valid.

    """
    from torch.optim import __all__

    optim_config = optim_config or dict()

    if optim_name not in __all__:
        raise ValueError(f"Optimizer {0} is not a valid optimizer. Choose one from {1}".format(optim_name, __all__))

    opt = cast(Optimizer, getattr(__import__("torch.optim", fromlist=[optim_name]), optim_name)(params, **optim_config))
    return opt


def construct_scheduler(
    optimizer: Optimizer, scheduler_name: str, scheduler_config: Mapping[str, Any] | None = None
) -> LRScheduler | ReduceLROnPlateau | LRSchedulerConfig:
    """Retrieve the scheduler class.

    Args:
        optimizer (Optimizer):
            Optimizer to be used with the scheduler.
        scheduler_name (str):
            Name of the scheduler to be used.
        scheduler_config (dict[str, Any], optional):
            Additional keyword arguments to be passed to the scheduler. Defaults to an empty dictionary.

    Returns:
        LRScheduler | ReduceLROnPlateau | LRSchedulerConfig

    """
    from torch.optim.lr_scheduler import __all__

    scheduler_config = scheduler_config or dict()

    if scheduler_name not in __all__:
        raise ValueError(f"Scheduler {0} is not a valid scheduler. Choose one from {1}".format(scheduler_name, __all__))

    # Initialize lr_config to None
    lr_config = None

    # Check if 'lr_scheduler_config' is in scheduler_config
    if "lr_scheduler_config" in scheduler_config:
        lr_config = dict(scheduler_config.pop("lr_scheduler_config"))

    sched = cast(
        LRScheduler,
        getattr(__import__("torch.optim.lr_scheduler", fromlist=[scheduler_name]), scheduler_name)(
            optimizer, **scheduler_config
        ),
    )

    # Validate type of 'lr_scheduler_config'
    if lr_config is not None:
        # if isinstance(lr_config, Mapping):
        #     lr_config["scheduler"] = sched
        #     return LRSchedulerConfig(**lr_config)
        # else:
        #     raise ValueError("lr_scheduler_config must be a dictionary.")
        lr_config["scheduler"] = sched
        return cast(LRSchedulerConfig, lr_config)

    return sched


def construct_loss(loss_name: str, loss_kwargs: Mapping[str, Any] | None = None) -> _Loss:
    """Construct a PyTorch loss function.

    Args:
        loss_name (str):
            Name of the loss function to be used.
        loss_kwargs (dict[str, Any], optional):
            Additional keyword arguments to be passed to the loss function. Defaults to an empty dictionary.

    Returns:
        torch.nn.Module:
            An instance of the specified loss function.

    Raises:
        ValueError:
            If the specified loss name is not valid.

    """
    from torch.nn.modules.loss import __all__ as __tloss__

    from .loss import __all__ as __closs__

    __all__ = __closs__ + __tloss__
    loss_kwargs = loss_kwargs or dict()

    if loss_name not in __all__:
        raise ValueError(f"Loss {loss_name} is not a valid loss. Choose one from {__all__}")

    if loss_name in __tloss__:
        return cast(_Loss, getattr(__import__("torch.nn", fromlist=[loss_name]), loss_name)(**loss_kwargs))
    else:
        return cast(
            _Loss, getattr(__import__("adf.core.lightning.loss", fromlist=[loss_name]), loss_name)(**loss_kwargs)
        )
