"""Inference callbacks for the Dueling Double DQN agent."""

from __future__ import annotations

import math
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .features import ACTIONS, FEATURE_NAMES, MOVE_DELTAS, FeatureExtractor
from .model import DuelingQNetwork


MODEL_FILE = "dueling_ddqn_model.pt"
INPUT_DIM = len(ACTIONS) * len(FEATURE_NAMES)
EPSILON_START = 0.60
EPSILON_MIN = 0.05
EPSILON_DECAY_STEPS = 120_000.0
HUNT_CENTER_Q_BIAS = 0.00
REVISIT_Q_PENALTY = 0.60
REVISIT_HISTORY_LENGTH = 32
OBJECTIVE_Q_BIAS = 1.00


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
    self.recent_positions = deque(maxlen=REVISIT_HISTORY_LENGTH)
    self.position_round = None
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
    action_features = self.feature_extractor.action_features(game_state)
    position = tuple(game_state["self"][-1])
    if self.position_round != game_state["round"]:
        self.recent_positions.clear()
        self.position_round = game_state["round"]
    revisit_penalties = np.zeros(len(ACTIONS), dtype=np.float32)
    danger_index = FEATURE_NAMES.index("target_danger")
    hunt_phase = (
        not game_state["coins"]
        and not np.any(game_state["field"] == 1)
        and bool(game_state["others"])
    )
    if hunt_phase or not allowed[ACTIONS.index("BOMB")]:
        for index, action in enumerate(ACTIONS):
            if action == "BOMB" or action_features[index, danger_index] > 0.0:
                continue
            target = (
                position[0] + MOVE_DELTAS[action][0],
                position[1] + MOVE_DELTAS[action][1],
            )
            revisit_penalties[index] = REVISIT_Q_PENALTY * min(
                self.recent_positions.count(target), 3
            )
    self.recent_positions.append(position)

    if self.train and self.rng.random() < exploration_rate(self):
        return ACTIONS[self.rng.choice(allowed_indices.tolist())]

    state = torch.from_numpy(action_features.astype(np.float32).reshape(-1)).unsqueeze(0).to(self.device)
    with torch.no_grad():
        q_values = self.policy_net(state).squeeze(0).cpu().numpy()
    center_index = FEATURE_NAMES.index("hunt_center_progress")
    objective_index = FEATURE_NAMES.index("objective_target")
    q_values = (
        q_values
        + self.hunt_center_q_bias * action_features[:, center_index]
        + OBJECTIVE_Q_BIAS * action_features[:, objective_index]
        - revisit_penalties
    )
    masked_q = np.where(allowed, q_values, -np.inf)
    best = np.flatnonzero(np.isclose(masked_q, np.max(masked_q)))
    return ACTIONS[self.rng.choice(best.tolist())]
