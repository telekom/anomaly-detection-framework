# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import multiprocessing as mp

from collections.abc import Callable
from omegaconf import DictConfig, OmegaConf
from typing import TYPE_CHECKING, Any

from .utils import construct_optimizer

if TYPE_CHECKING:
    import optuna


def validate_optuna_cfg(cfg: DictConfig) -> None:
    """Validate that the Optuna config block exists and has required keys.

    Args:
        cfg: Hydra/OmegaConf config expected to contain the `optuna` section.

    """
    if getattr(cfg, "optuna", None) is None:
        raise ValueError("Missing required config: optuna")
    if "n_trials" not in cfg.optuna:
        raise ValueError("Missing required config: optuna.n_trials")
    if "n_jobs" not in cfg.optuna:
        raise ValueError("Missing required config: optuna.n_jobs")
    if "n_workers" not in cfg.optuna:
        raise ValueError("Missing required config: optuna.n_workers")


def create_study_from_cfg(cfg: DictConfig) -> optuna.Study:
    """Create an Optuna study from the config.

    Args:
        cfg: Hydra/OmegaConf config containing the `optuna` section.

    Returns:
        The created Optuna study.

    """
    import optuna

    validate_optuna_cfg(cfg)
    optuna_cfg = cfg.optuna
    direction = optuna_cfg.get("direction", "maximize")
    study_name = optuna_cfg.get("study_name", "optuna_study")
    storage = optuna_cfg.get("storage", None)

    logging.info("[Optuna] Config direction=%s study=%s storage=%s", direction, study_name, storage)
    return optuna.create_study(
        direction=direction, study_name=study_name, storage=storage, load_if_exists=bool(storage)
    )


def run_optuna_study(
    cfg: DictConfig,
    objective: Callable[[optuna.Trial, DictConfig], float],
    n_trials: int | None = None,
    n_jobs: int | None = None,
) -> optuna.Study:
    """Run an Optuna study in the current process.

    Args:
        cfg: Hydra/OmegaConf config containing the `optuna` section.
        objective: Callable that receives (trial, cfg) and returns a scalar metric.
        n_trials: Optional override for the number of trials.
        n_jobs: Optional override for the number of parallel jobs within a process.

    Returns:
        The optimized Optuna study.

    """
    validate_optuna_cfg(cfg)
    optuna_cfg = cfg.optuna
    if n_trials is None:
        n_trials = int(optuna_cfg.get("n_trials"))
    if n_jobs is None:
        n_jobs = int(optuna_cfg.get("n_jobs"))

    study = create_study_from_cfg(cfg)

    def wrapped_objective(trial: optuna.Trial) -> float:
        return objective(trial, cfg)

    logging.info("[Optuna] Running study n_trials=%s n_jobs=%s", n_trials, n_jobs)
    study.optimize(wrapped_objective, n_trials=n_trials, n_jobs=n_jobs)
    return study


def run_optuna_multiprocess(cfg: DictConfig, worker_fn: Callable[[dict[str, Any], int], None]) -> bool:
    """Run Optuna in multiple OS processes.

    This is a lightweight orchestrator: it spawns `optuna.n_workers` processes and
    delegates trial execution to `worker_fn`. Storage must be configured so workers
    share the same study state.

    Args:
        cfg: Hydra/OmegaConf config containing the `optuna` section.
        worker_fn: Worker entrypoint called as `worker_fn(cfg_payload, worker_id)`.

    Returns:
        True if multiprocessing was used, False if `optuna.n_workers <= 1`.

    """
    validate_optuna_cfg(cfg)
    optuna_cfg = cfg.optuna
    n_workers = int(optuna_cfg.get("n_workers"))
    storage = optuna_cfg.get("storage")

    if n_workers <= 1:
        return False
    if not storage:
        raise ValueError("optuna.storage must be set when optuna.n_workers > 1")

    # Reason: initialize Optuna RDB schema once in the parent process.
    # This avoids concurrent Alembic stamping/migrations in multiple workers,
    # which can fail with SQLite UNIQUE constraint errors on `alembic_version`.
    _ = create_study_from_cfg(cfg)

    cfg_payload = OmegaConf.to_container(cfg, resolve=True)
    # Reason: spawn isolated workers for parallel trials.
    processes: list[mp.Process] = []
    for worker_id in range(n_workers):
        proc = mp.Process(target=worker_fn, args=(cfg_payload, worker_id), name=f"optuna-worker-{worker_id}")
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()
    return True


def run_lr_finder(
    model: Any,
    train_dataloader: Any,
    optimizer_name: str,
    optimizer_config: dict[str, Any] | None = None,
    start_lr: float = 1e-6,
    end_lr: float = 1e-1,
    num_steps: int = 200,
    stop_on_diverge: bool = True,
    diverge_factor: float = 4.0,
    logger: Any | None = None,
    step_offset: int = 0,
) -> float:
    """Run a Leslie Smith LR range test and return the suggested LR.

    Ramps LR exponentially from start_lr to end_lr over num_steps (no scheduler,
    no one-cycle). Cycles through the dataloader to guarantee the full ramp.
    Stops when loss diverges. Returns the LR at the steepest decline in smoothed loss.

    If logger has an ``experiment`` attr (e.g. Lightning TensorBoardLogger),
    logs ``lr_finder/loss``, ``lr_finder/smoothed_loss``, ``lr_finder/lr`` per step.
    Use ``step_offset`` to align LR finder steps with training steps in TensorBoard.
    """
    import math
    import torch

    from itertools import cycle

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    model.train()
    optimizer_config = dict(optimizer_config or {})
    optimizer_config["lr"] = start_lr
    optimizer = construct_optimizer(model.parameters(), optimizer_name, optimizer_config)

    log_start = math.log10(start_lr)
    log_end = math.log10(end_lr)
    step_size = (log_end - log_start) / max(1, num_steps - 1)

    best_loss = float("inf")
    smoothed = None
    beta = 0.98
    history: list[tuple[float, float, float]] = []
    writer = getattr(logger, "experiment", None) if logger else None

    def _to_device(item: Any) -> Any:
        if torch.is_tensor(item):
            return item.to(device)
        if isinstance(item, (list, tuple)):
            return type(item)(_to_device(x) for x in item)
        if isinstance(item, dict):
            return {k: _to_device(v) for k, v in item.items()}
        return item

    dataloader_iter = cycle(train_dataloader)
    for step in range(num_steps):
        batch = _to_device(next(dataloader_iter))
        optimizer.zero_grad(set_to_none=True)

        lr = 10 ** (log_start + step * step_size)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if isinstance(batch, (tuple, list)):
            inputs = batch[0]
        else:
            inputs = batch

        outputs = model(inputs)
        if hasattr(model, "loss") and callable(model.loss):
            if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
                loss = model.loss(outputs[0], outputs[1])
            else:
                target = inputs
                if isinstance(batch, (tuple, list)) and len(batch) > 1:
                    target = batch[1]
                try:
                    loss = model.loss(outputs, target)
                except TypeError:
                    loss = model.loss(outputs, inputs)
        else:
            raise ValueError("run_lr_finder requires model.loss(...) to be implemented.")
        if torch.is_tensor(loss) and loss.ndim > 0:
            loss = loss.mean()
        loss_value = float(loss.item())

        smoothed = loss_value if smoothed is None else beta * smoothed + (1 - beta) * loss_value
        corrected = smoothed / (1 - beta ** (step + 1))
        history.append((lr, loss_value, corrected))

        if writer is not None:
            global_step = step_offset + step
            writer.add_scalar("lr_finder/loss", loss_value, global_step)
            writer.add_scalar("lr_finder/smoothed_loss", corrected, global_step)
            writer.add_scalar("lr_finder/lr", lr, global_step)
            writer.add_scalar(f"lr-{optimizer_name}", lr, global_step)

        if corrected < best_loss:
            best_loss = corrected

        if stop_on_diverge and corrected > diverge_factor * best_loss:
            logging.info("[LR finder] diverged at step=%d lr=%.2e", step, lr)
            break

        loss.backward()
        optimizer.step()

    if len(history) < 2:
        best_lr = history[0][0] if history else start_lr
    else:
        steepest_idx = 1
        steepest_drop = -float("inf")
        for i in range(1, len(history)):
            drop = history[i - 1][2] - history[i][2]
            if drop > steepest_drop:
                steepest_drop = drop
                steepest_idx = i
        best_lr = history[steepest_idx][0]

    logging.info("[LR finder] best_lr=%.2e (at steepest decline) total_steps=%d", best_lr, len(history))
    return best_lr
