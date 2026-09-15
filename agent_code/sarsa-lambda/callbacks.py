"""Inference and on-policy action selection for linear SARSA(lambda)."""

from __future__ import annotations

import math
import random
import json
from pathlib import Path

import numpy as np

from .features import ACTIONS, FeatureExtractor
from .model import SarsaLambdaModel


MODEL_FILE = "sarsa_lambda_model.npz"
EPSILON_START = 1.0
EPSILON_MIN = 0.03
EPSILON_DECAY_EPISODES = 1200.0


def setup(self):
    parameter_path = Path("parameters.json")
    self.parameters = json.loads(parameter_path.read_text()) if parameter_path.exists() else {}
    self.feature_extractor = FeatureExtractor()
    self.model = SarsaLambdaModel.load(MODEL_FILE, self.feature_extractor.feature_count)
    self.rng = random.Random()
    self.pending_transition = None
    self.logger.info(
        "Loaded SARSA(lambda): episodes=%d updates=%d", self.model.episodes, self.model.updates
    )


def exploration_rate(self):
    if not self.train:
        return 0.0
    epsilon_start = float(self.parameters.get("epsilon_start", EPSILON_START))
    epsilon_min = float(self.parameters.get("epsilon_min", EPSILON_MIN))
    decay = float(self.parameters.get("epsilon_decay_episodes", EPSILON_DECAY_EPISODES))
    schedule_start = int(self.parameters.get("schedule_start_episode", 0))
    elapsed = max(0, self.model.episodes - schedule_start)
    return epsilon_min + (epsilon_start - epsilon_min) * math.exp(
        -elapsed / decay
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


def act(self, game_state: dict) -> str:
    action = _select_action(self, game_state)
    # The framework calls game_events_occurred before this act. Therefore this
    # is the first point at which the actual behaviour-policy a_(t+1) exists.
    if self.train and self.pending_transition is not None:
        from .train import complete_pending_update

        complete_pending_update(self, game_state, action)
    return action


def state_to_features(game_state):
    return None if game_state is None else FeatureExtractor().action_features(game_state)
