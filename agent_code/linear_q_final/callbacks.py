"""Configurable inference callback for the final linear-Q experiment."""

import json
import math
import random
from pathlib import Path

import numpy as np

from .features import ACTIONS, FeatureExtractor
from .model import LinearQModel

MODEL_FILE = "linear_q_model.npz"


def setup(self):
    path = Path("parameters.json")
    self.parameters = json.loads(path.read_text()) if path.exists() else {}
    self.feature_extractor = FeatureExtractor()
    self.model = LinearQModel.load(MODEL_FILE, self.feature_extractor.feature_count)
    self.rng = random.Random()


def exploration_rate(self):
    if not self.train:
        return 0.0
    start = float(self.parameters.get("epsilon_start", 0.12))
    minimum = float(self.parameters.get("epsilon_min", 0.01))
    decay = float(self.parameters.get("epsilon_decay_updates", 50000.0))
    offset = int(self.parameters.get("schedule_start_updates", 61973))
    elapsed = max(0, self.model.updates - offset)
    return minimum + (start - minimum) * math.exp(-elapsed / decay)


def act(self, game_state):
    matrix = self.feature_extractor.action_features(game_state)
    allowed = self.feature_extractor.safe_action_mask(game_state)
    indices = np.flatnonzero(allowed)
    if indices.size == 0:
        return "WAIT"
    if self.train and self.rng.random() < exploration_rate(self):
        return ACTIONS[self.rng.choice(indices.tolist())]
    values = np.where(allowed, self.model.q_values(matrix), -np.inf)
    best = np.flatnonzero(np.isclose(values, np.max(values)))
    return ACTIONS[self.rng.choice(best.tolist())]


def state_to_features(game_state):
    return None if game_state is None else FeatureExtractor().action_features(game_state)
