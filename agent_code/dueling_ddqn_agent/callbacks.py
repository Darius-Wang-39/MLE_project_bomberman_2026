"""Inference callbacks for the Dueling Double DQN agent."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch

from .features import ACTIONS, FEATURE_NAMES, FeatureExtractor
from .model import DuelingQNetwork


MODEL_FILE = "dueling_ddqn_model.pt"
INPUT_DIM = len(ACTIONS) * len(FEATURE_NAMES)
EPSILON_START = 0.60
EPSILON_MIN = 0.05
EPSILON_DECAY_STEPS = 120_000.0


def vectorize_state(extractor: FeatureExtractor, game_state: dict) -> np.ndarray:
    return extractor.action_features(game_state).astype(np.float32).reshape(-1)


def setup(self):
    self.device = torch.device("cuda" if self.train and torch.cuda.is_available() else "cpu")
    self.feature_extractor = FeatureExtractor()
    self.policy_net = DuelingQNetwork(INPUT_DIM, len(ACTIONS)).to(self.device)
    self.environment_steps = 0
    self.learner_updates = 0
    self.loaded_optimizer_state = None
    checkpoint_path = Path(MODEL_FILE)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_state"])
        self.environment_steps = int(checkpoint.get("environment_steps", 0))
        self.learner_updates = int(checkpoint.get("learner_updates", 0))
        self.loaded_optimizer_state = checkpoint.get("optimizer_state")
    self.policy_net.eval()
    self.rng = random.Random()
    self.logger.info(
        "Loaded Dueling DDQN on %s at env_step=%d learner_update=%d",
        self.device,
        self.environment_steps,
        self.learner_updates,
    )


def exploration_rate(self) -> float:
    if not self.train:
        return 0.0
    return EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * math.exp(
        -self.environment_steps / EPSILON_DECAY_STEPS
    )


def act(self, game_state: dict) -> str:
    allowed = self.feature_extractor.safe_action_mask(game_state)
    allowed_indices = np.flatnonzero(allowed)
    if allowed_indices.size == 0:
        return "WAIT"
    if self.train and self.rng.random() < exploration_rate(self):
        return ACTIONS[self.rng.choice(allowed_indices.tolist())]

    state = torch.from_numpy(vectorize_state(self.feature_extractor, game_state)).unsqueeze(0).to(self.device)
    with torch.no_grad():
        q_values = self.policy_net(state).squeeze(0).cpu().numpy()
    masked_q = np.where(allowed, q_values, -np.inf)
    best = np.flatnonzero(np.isclose(masked_q, np.max(masked_q)))
    return ACTIONS[self.rng.choice(best.tolist())]

