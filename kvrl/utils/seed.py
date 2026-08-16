"""Deterministic seeding for numpy / torch / python."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def np_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
