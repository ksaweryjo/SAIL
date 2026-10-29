import os
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def _atomic_save(state: dict[str, Any], path: str | os.PathLike) -> None:
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def save_checkpoint(
    state: dict[str, Any],
    checkpoint_dir: str | os.PathLike,
    config_name: str = "baseline_config",
    filename: str | None = None,
    is_best: bool = False,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)

    if filename is None:
        filename = f"{config_name}_last.pth"

    _atomic_save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        _atomic_save(
            state, os.path.join(checkpoint_dir, f"{config_name}_model_best.pth")
        )


def load_checkpoint(
    checkpoint_path: str | os.PathLike,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> tuple[int, float]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    best_val_loss = checkpoint.get("best_val_loss", checkpoint["val_loss"])
    return checkpoint["epoch"], best_val_loss


def read_wandb_run_id(checkpoint_path: str | os.PathLike | None) -> str | None:
    if checkpoint_path is None:
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    return checkpoint.get("wandb_run_id")