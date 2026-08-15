"""Small serializable linear action-value model."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class LinearQModel:
    def __init__(self, feature_count: int, weights: np.ndarray | None = None, updates: int = 0):
        self.weights = np.zeros(feature_count, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
        self.updates = int(updates)

    def q_values(self, feature_matrix: np.ndarray) -> np.ndarray:
        return np.asarray(feature_matrix, dtype=np.float64) @ self.weights

    def update(
        self,
        features: np.ndarray,
        target: float,
        alpha: float,
        l2: float = 1e-5,
        td_clip: float = 20.0,
    ) -> float:
        prediction = float(np.dot(features, self.weights))
        td_error = float(np.clip(target - prediction, -td_clip, td_clip))
        self.weights *= 1.0 - alpha * l2
        self.weights += alpha * td_error * features
        self.updates += 1
        return td_error

    def save(self, path: str | Path) -> None:
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(temporary, weights=self.weights, updates=np.array([self.updates], dtype=np.int64))
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: str | Path, feature_count: int) -> "LinearQModel":
        path = Path(path)
        if not path.exists():
            return cls(feature_count)
        with np.load(path) as data:
            weights = np.asarray(data["weights"], dtype=np.float64)
            updates = int(np.asarray(data["updates"]).reshape(-1)[0])
        if weights.shape != (feature_count,):
            raise ValueError(f"Model has {weights.shape[0]} weights, expected {feature_count}")
        return cls(feature_count, weights=weights, updates=updates)

