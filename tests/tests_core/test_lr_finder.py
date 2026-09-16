# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

import torch

from torch.utils.data import DataLoader, TensorDataset

from adf.core.lightning.hpo import run_lr_finder


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def loss(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(preds, target)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, y = batch
        preds = self.linear(x)
        return self.loss_fn(preds, y)


def test_run_lr_finder_returns_value() -> None:
    torch.manual_seed(0)
    x = torch.randn(16, 4)
    y = torch.randn(16, 1)
    dataloader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    model = _DummyModel()

    lr = run_lr_finder(
        model,
        dataloader,
        optimizer_name="AdamW",
        optimizer_config={"weight_decay": 0.0},
        start_lr=1e-6,
        end_lr=1e-3,
        num_steps=5,
    )
    assert 1e-6 <= lr <= 1e-3


class _DummyWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, name: str, value: float, step: int) -> None:
        self.scalars.append((name, float(value), int(step)))


class _DummyLogger:
    def __init__(self) -> None:
        self.experiment = _DummyWriter()


class _ScriptedLossModel(torch.nn.Module):
    def __init__(self, losses: list[float]) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))
        self.losses = losses
        self.loss_idx = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def loss(self, _preds: torch.Tensor, _target: torch.Tensor) -> torch.Tensor:
        value = self.losses[min(self.loss_idx, len(self.losses) - 1)]
        self.loss_idx += 1
        return self.anchor * 0 + torch.tensor(value, device=self.anchor.device)


class _ModelWithExistingOneCycle(torch.nn.Module):
    """Model with a preconfigured optimizer/scheduler used only for isolation checks."""

    def __init__(self, total_steps: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
        self.loss_fn = torch.nn.MSELoss()
        self.existing_optimizer = torch.optim.SGD(self.parameters(), lr=0.2)
        self.existing_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.existing_optimizer,
            max_lr=0.5,
            total_steps=total_steps,
            pct_start=0.5,
            div_factor=5.0,
            final_div_factor=10.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def loss(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(preds, target)


def _expected_exponential_lrs(start_lr: float, end_lr: float, num_steps: int) -> list[float]:
    """Build the LR-finder exponential sweep sequence."""
    log_start = torch.log10(torch.tensor(start_lr)).item()
    log_end = torch.log10(torch.tensor(end_lr)).item()
    step_size = (log_end - log_start) / max(1, num_steps - 1)
    return [10 ** (log_start + step * step_size) for step in range(num_steps)]


def _one_cycle_trace(total_steps: int) -> list[float]:
    """Build a reference OneCycle LR trace for comparison."""
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=0.2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.5,
        total_steps=total_steps,
        pct_start=0.5,
        div_factor=5.0,
        final_div_factor=10.0,
    )
    trace: list[float] = []
    for _ in range(total_steps):
        trace.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        scheduler.step()
    return trace


def _expected_best_lr(start_lr: float, end_lr: float, losses: list[float], num_steps: int) -> float:
    beta = 0.98
    diverge_factor = 4.0
    log_start = torch.log10(torch.tensor(start_lr)).item()
    log_end = torch.log10(torch.tensor(end_lr)).item()
    step_size = (log_end - log_start) / max(1, num_steps - 1)

    history: list[tuple[float, float]] = []
    best_smoothed = float("inf")
    smoothed: float | None = None
    for step in range(num_steps):
        lr = 10 ** (log_start + step * step_size)
        loss_value = losses[min(step, len(losses) - 1)]
        smoothed = loss_value if smoothed is None else beta * smoothed + (1 - beta) * loss_value
        corrected = smoothed / (1 - beta ** (step + 1))
        history.append((lr, corrected))
        if corrected < best_smoothed:
            best_smoothed = corrected
        if corrected > diverge_factor * best_smoothed:
            break

    steepest_idx = 1
    steepest_drop = -float("inf")
    for i in range(1, len(history)):
        drop = history[i - 1][1] - history[i][1]
        if drop > steepest_drop:
            steepest_drop = drop
            steepest_idx = i
    return history[steepest_idx][0]


def test_run_lr_finder_selects_steepest_drop_and_stops_on_divergence() -> None:
    # Decreasing losses followed by a strong spike to force divergence stop.
    scripted_losses = [8.0, 7.5, 7.1, 6.8, 6.6, 6.4, 6.0, 5.2, 4.0, 3.4, 3.0, 2000.0]
    num_steps = 20
    start_lr = 1e-5
    end_lr = 1e-2
    model = _ScriptedLossModel(scripted_losses)
    dataloader = DataLoader(TensorDataset(torch.randn(4, 4), torch.randn(4, 4)), batch_size=2)
    logger = _DummyLogger()

    best_lr = run_lr_finder(
        model=model,
        train_dataloader=dataloader,
        optimizer_name="AdamW",
        optimizer_config={"weight_decay": 0.0},
        start_lr=start_lr,
        end_lr=end_lr,
        num_steps=num_steps,
        logger=logger,
        step_offset=0,
    )

    expected_lr = _expected_best_lr(start_lr, end_lr, scripted_losses, num_steps)
    assert torch.isclose(torch.tensor(best_lr), torch.tensor(expected_lr), atol=1e-12, rtol=0.0)

    lr_points = [value for name, value, _ in logger.experiment.scalars if name == "lr_finder/lr"]
    # Divergence should stop the sweep early.
    assert 2 <= len(lr_points) < num_steps
    assert all(lr_points[i] < lr_points[i + 1] for i in range(len(lr_points) - 1))


def test_run_lr_finder_ignores_existing_onecycle_scheduler() -> None:
    """LR finder must use its own sweep even if model already has OneCycle scheduler."""
    torch.manual_seed(0)
    num_steps = 8
    start_lr = 1e-5
    end_lr = 1e-2
    dataloader = DataLoader(
        TensorDataset(torch.randn(16, 4), torch.randn(16, 1)),
        batch_size=4,
        shuffle=False,
    )
    model = _ModelWithExistingOneCycle(total_steps=num_steps)
    logger = _DummyLogger()
    original_model_optimizer_lr = float(model.existing_optimizer.param_groups[0]["lr"])
    original_model_scheduler_epoch = int(model.existing_scheduler.last_epoch)

    _ = run_lr_finder(
        model=model,
        train_dataloader=dataloader,
        optimizer_name="AdamW",
        optimizer_config={"weight_decay": 0.0},
        start_lr=start_lr,
        end_lr=end_lr,
        num_steps=num_steps,
        stop_on_diverge=False,
        logger=logger,
    )

    lr_points = [value for name, value, _ in logger.experiment.scalars if name == "lr_finder/lr"]
    assert len(lr_points) == num_steps
    expected_exp_lrs = _expected_exponential_lrs(start_lr, end_lr, num_steps)
    for observed_lr, expected_lr in zip(lr_points, expected_exp_lrs, strict=True):
        assert torch.isclose(torch.tensor(observed_lr), torch.tensor(expected_lr), atol=1e-12, rtol=0.0)

    expected_one_cycle = _one_cycle_trace(total_steps=num_steps)
    assert not all(
        torch.isclose(torch.tensor(observed), torch.tensor(oc), atol=1e-12, rtol=0.0)
        for observed, oc in zip(lr_points, expected_one_cycle, strict=True)
    )

    # Existing training optimizer/scheduler attached to model must remain untouched.
    assert torch.isclose(
        torch.tensor(model.existing_optimizer.param_groups[0]["lr"]),
        torch.tensor(original_model_optimizer_lr),
        atol=1e-12,
        rtol=0.0,
    )
    assert model.existing_scheduler.last_epoch == original_model_scheduler_epoch
