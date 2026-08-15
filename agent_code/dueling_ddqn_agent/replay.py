"""A small proportional prioritized replay buffer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_mask: np.ndarray
    done: bool
    discount: float


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.storage = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float64)
        self.position = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, transition: Transition):
        priority = float(self.priorities[: self.size].max()) if self.size else 1.0
        self.storage[self.position] = transition
        self.priorities[self.position] = max(priority, 1e-6)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float, rng: np.random.Generator):
        if self.size < batch_size:
            raise ValueError("Not enough transitions for a full batch")
        scaled = np.power(self.priorities[: self.size], self.alpha)
        probabilities = scaled / scaled.sum()
        indices = rng.choice(self.size, size=batch_size, replace=False, p=probabilities)
        weights = np.power(self.size * probabilities[indices], -beta)
        weights /= weights.max()
        transitions = [self.storage[int(index)] for index in indices]
        return transitions, indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            self.priorities[int(index)] = max(float(priority), 1e-6)

