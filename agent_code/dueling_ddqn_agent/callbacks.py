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
HUNT_CENTER_Q_BIAS = 0.08


def migrate_policy_input(policy_state, target_state):
    """Expand action-row features while preserving every learned old weight."""
    key = "encoder.0.weight"
    old_weight = policy_state[key]
    target_weight = target_state[key]
    if old_weight.shape == target_weight.shape:
        return policy_state, False
    if old_weight.shape[0] != target_weight.shape[0]:
        raise ValueError("Cannot migrate a checkpoint with a different hidden width")
    if old_weight.shape[1] % len(ACTIONS) or target_weight.shape[1] % len(ACTIONS):
        raise ValueError("Input widths must contain one equal feature row per action")

    old_feature_count = old_weight.shape[1] // len(ACTIONS)
    new_feature_count = target_weight.shape[1] // len(ACTIONS)
    if new_feature_count <= old_feature_count:
        raise ValueError("Automatic migration only supports appended action features")

    migrated_weight = torch.zeros_like(target_weight)
    for action_index in range(len(ACTIONS)):
        old_start = action_index * old_feature_count
        new_start = action_index * new_feature_count
        migrated_weight[:, new_start : new_start + old_feature_count] = old_weight[
            :, old_start : old_start + old_feature_count
        ]
    migrated = dict(policy_state)
    migrated[key] = migrated_weight
    return migrated, True


def vectorize_state(extractor: FeatureExtractor, game_state: dict) -> np.ndarray:
    return extractor.action_features(game_state).astype(np.float32).reshape(-1)


def setup(self):
    self.device = torch.device("cuda" if self.train and torch.cuda.is_available() else "cpu")
    self.feature_extractor = FeatureExtractor()
    self.policy_net = DuelingQNetwork(INPUT_DIM, len(ACTIONS)).to(self.device)
    self.environment_steps = 0
    self.learner_updates = 0
    self.loaded_optimizer_state = None
    self.hunt_center_q_bias = HUNT_CENTER_Q_BIAS
    checkpoint_path = Path(MODEL_FILE)
    self.checkpoint_migrated = False
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        policy_state, self.checkpoint_migrated = migrate_policy_input(
            checkpoint["policy_state"], self.policy_net.state_dict()
        )
        self.policy_net.load_state_dict(policy_state)
        self.environment_steps = int(checkpoint.get("environment_steps", 0))
        self.learner_updates = int(checkpoint.get("learner_updates", 0))
        self.loaded_optimizer_state = (
            None if self.checkpoint_migrated else checkpoint.get("optimizer_state")
        )
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

    action_features = self.feature_extractor.action_features(game_state)
    state = torch.from_numpy(action_features.astype(np.float32).reshape(-1)).unsqueeze(0).to(self.device)
    with torch.no_grad():
        q_values = self.policy_net(state).squeeze(0).cpu().numpy()
    center_index = FEATURE_NAMES.index("hunt_center_progress")
    q_values = q_values + self.hunt_center_q_bias * action_features[:, center_index]
    masked_q = np.where(allowed, q_values, -np.inf)
    best = np.flatnonzero(np.isclose(masked_q, np.max(masked_q)))
    return ACTIONS[self.rng.choice(best.tolist())]
