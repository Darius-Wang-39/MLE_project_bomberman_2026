"""Evaluation callbacks for the linear Q-learning agent."""

from __future__ import annotations

import math
import random

import numpy as np

from .features import ACTIONS, FeatureExtractor
from .model import LinearQModel


MODEL_FILE = "linear_q_model.npz"
EPSILON_START = 0.35
EPSILON_MIN = 0.03
EPSILON_DECAY_UPDATES = 40_000.0


def setup(self):
    self.feature_extractor = FeatureExtractor()
    self.model = LinearQModel.load(MODEL_FILE, self.feature_extractor.feature_count)
    self.rng = random.Random()
    self.logger.info("Loaded linear Q agent with %d updates", self.model.updates)


def exploration_rate(self) -> float:
    if not self.train:
        return 0.0
    return EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * math.exp(
        -self.model.updates / EPSILON_DECAY_UPDATES
    )


def act(self, game_state: dict) -> str:
    feature_matrix = self.feature_extractor.action_features(game_state)
    allowed = self.feature_extractor.safe_action_mask(game_state)
    allowed_indices = np.flatnonzero(allowed)
    if allowed_indices.size == 0:
        return "WAIT"

    epsilon = exploration_rate(self)
    if self.train and self.rng.random() < epsilon:
        return ACTIONS[self.rng.choice(allowed_indices.tolist())]

    q_values = self.model.q_values(feature_matrix)
    masked_q = np.where(allowed, q_values, -np.inf)
    best_value = np.max(masked_q)
    best_indices = np.flatnonzero(np.isclose(masked_q, best_value))
    return ACTIONS[self.rng.choice(best_indices.tolist())]


def state_to_features(game_state: dict):
    if game_state is None:
        return None
    return FeatureExtractor().action_features(game_state)

