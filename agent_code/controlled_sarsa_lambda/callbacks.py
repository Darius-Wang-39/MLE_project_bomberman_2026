from __future__ import annotations

import math
import importlib

import numpy as np

from agent_code.controlled_common.features import ACTIONS, FeatureExtractor
from agent_code.controlled_common.seeding import python_rng
SarsaLambdaModel = importlib.import_module(
    "agent_code.sarsa-lambda.model"
).SarsaLambdaModel

MODEL_FILE = "model.npz"
EPSILON_START = 1.0
EPSILON_MIN = 0.03
EPSILON_DECAY_EPISODES = 1200.0


def setup(self):
    self.parameters = {}
    self.feature_extractor = FeatureExtractor()
    self.model = SarsaLambdaModel.load(MODEL_FILE, self.feature_extractor.feature_count)
    self.rng = python_rng(21)
    self.pending_transition = None


def exploration_rate(self):
    if not self.train:
        return 0.0
    return EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * math.exp(
        -self.model.episodes / EPSILON_DECAY_EPISODES
    )


def _select_action(self, game_state):
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


def act(self, game_state):
    action = _select_action(self, game_state)
    if self.train and self.pending_transition is not None:
        from .train import complete_pending_update

        complete_pending_update(self, game_state, action)
    return action


def state_to_features(game_state):
    return None if game_state is None else FeatureExtractor().action_features(game_state)
