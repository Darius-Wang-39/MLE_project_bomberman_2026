"""Separated reproducible RNG streams for the controlled experiment."""

from __future__ import annotations

import os
import random

import numpy as np


def experiment_seed() -> int:
    return int(os.environ.get("CONTROLLED_AGENT_SEED", "0"))


def python_rng(offset: int = 0) -> random.Random:
    return random.Random(experiment_seed() + offset)


def numpy_rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(experiment_seed() + offset)


def seed_torch(offset: int = 0) -> int:
    seed = experiment_seed() + offset
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
