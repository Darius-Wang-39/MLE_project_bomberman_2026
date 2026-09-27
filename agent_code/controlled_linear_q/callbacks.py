from __future__ import annotations

import math

import numpy as np

from agent_code.controlled_common.features import ACTIONS, FeatureExtractor
from agent_code.controlled_common.seeding import python_rng
from agent_code.linear_q_agent.model import LinearQModel

MODEL_FILE = "model.npz"
EPSILON_START = 0.35
EPSILON_MIN = 0.03
EPSILON_DECAY_UPDATES = 40_000.0


def setup(self):
    self.feature_extractor = FeatureExtractor()
    self.model = LinearQModel.load(MODEL_FILE, self.feature_extractor.feature_count)
    self.rng = python_rng(11)


def exploration_rate(self):
    if not self.train:
        return 0.0
    return EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * math.exp(
        -self.model.updates / EPSILON_DECAY_UPDATES
    )


def act(self, game_state):
    features = self.feature_extractor.action_features(game_state)
    allowed = self.feature_extractor.safe_action_mask(game_state)
    indices = np.flatnonzero(allowed)
    if indices.size == 0:
        return "WAIT"
    if self.train and self.rng.random() < exploration_rate(self):
        return ACTIONS[self.rng.choice(indices.tolist())]
    values = np.where(allowed, self.model.q_values(features), -np.inf)
    best = np.flatnonzero(np.isclose(values, np.max(values)))
    return ACTIONS[self.rng.choice(best.tolist())]


def state_to_features(game_state):
    return None if game_state is None else FeatureExtractor().action_features(game_state)
