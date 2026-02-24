"""Shared utility helpers: seeding, device selection, timestamps."""

from __future__ import annotations

__all__ = ["set_seed", "get_device", "timestamp", "ensure_parent"]

import random
from pathlib import Path
import time

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
