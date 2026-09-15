"""Serializable linear action-value model for SARSA(lambda)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class SarsaLambdaModel:
    def __init__(self, feature_count: int, weights=None, updates: int = 0, episodes: int = 0):
        self.weights = (
            np.zeros(feature_count, dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        self.updates = int(updates)
        self.episodes = int(episodes)

    def q_values(self, feature_matrix):
        return np.asarray(feature_matrix, dtype=np.float64) @ self.weights

    def save(self, path):
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            weights=self.weights,
            updates=np.array([self.updates], dtype=np.int64),
            episodes=np.array([self.episodes], dtype=np.int64),
        )
        os.replace(temporary, path)

    @classmethod
    def load(cls, path, feature_count):
        path = Path(path)
        if not path.exists():
            return cls(feature_count)
        with np.load(path) as data:
            weights = np.asarray(data["weights"], dtype=np.float64)
            updates = int(np.asarray(data["updates"]).reshape(-1)[0])
            episodes = int(np.asarray(data.get("episodes", [0])).reshape(-1)[0])
        if weights.shape != (feature_count,):
            raise ValueError(f"Model has shape {weights.shape}, expected ({feature_count},)")
        return cls(feature_count, weights, updates, episodes)
