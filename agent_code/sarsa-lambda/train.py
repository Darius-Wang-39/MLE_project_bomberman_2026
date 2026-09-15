"""True online-order SARSA(lambda) updates (accumulating eligibility traces)."""

from __future__ import annotations

from typing import List

import numpy as np

import events as e
from .callbacks import ACTIONS, MODEL_FILE, exploration_rate
from .features import FEATURE_NAMES


GAMMA = 0.95
LAMBDA = 0.70
ALPHA = 0.003
L2 = 1e-5
TD_CLIP = 20.0
SHAPING_SCALE = 0.5
BOMB_ESCAPE_REWARD = 0.35
UNSAFE_BOMB_PENALTY = -2.0

EVENT_REWARDS = {
    e.MOVED_LEFT: -0.02, e.MOVED_RIGHT: -0.02, e.MOVED_UP: -0.02, e.MOVED_DOWN: -0.02,
    e.WAITED: -0.05, e.INVALID_ACTION: -2.0, e.BOMB_DROPPED: -0.02,
    e.CRATE_DESTROYED: 0.8, e.COIN_FOUND: 0.5, e.COIN_COLLECTED: 4.0,
    e.KILLED_OPPONENT: 10.0, e.KILLED_SELF: -20.0, e.GOT_KILLED: -15.0,
    e.SURVIVED_ROUND: 2.0,
}


def setup_training(self):
    self.gamma = float(self.parameters.get("gamma", GAMMA))
    self.lambda_ = float(self.parameters.get("lambda", LAMBDA))
    self.alpha = float(self.parameters.get("alpha", ALPHA))
    self.bomb_escape_reward = float(self.parameters.get("bomb_escape_reward", BOMB_ESCAPE_REWARD))
    self.unsafe_bomb_penalty = float(self.parameters.get("unsafe_bomb_penalty", UNSAFE_BOMB_PENALTY))
    self.eligibility = np.zeros_like(self.model.weights)
    self.pending_transition = None
    self.last_td_error = 0.0
    self.episode_td_errors = []


def reward_from_events(self, events: List[str]):
    reward = float(sum(EVENT_REWARDS.get(event, 0.0) for event in events))
    self.logger.info("Awarded %.3f for events %s", reward, ", ".join(events))
    return reward


def _bomb_escape_shaping(self, old_state, action):
    if action != "BOMB":
        return 0.0
    row = self.feature_extractor.action_features(old_state)[ACTIONS.index("BOMB")]
    names = FEATURE_NAMES
    can_escape = row[names.index("bomb_escape")] > 0.0
    escape_space = row[names.index("escape_space")]
    if not can_escape:
        return self.unsafe_bomb_penalty
    return self.bomb_escape_reward * (0.5 + 0.5 * escape_space)


def _transition_reward(self, old_state, action, new_state, events):
    reward = reward_from_events(self, events)
    return reward + SHAPING_SCALE * (
        self.gamma * self.feature_extractor.state_potential(new_state)
        - self.feature_extractor.state_potential(old_state)
    ) + _bomb_escape_shaping(self, old_state, action)


def _update(self, features, reward, next_features=None):
    gamma = getattr(self, "gamma", GAMMA)
    lambda_ = getattr(self, "lambda_", LAMBDA)
    alpha = getattr(self, "alpha", ALPHA)
    current_q = float(np.dot(self.model.weights, features))
    next_q = 0.0 if next_features is None else float(np.dot(self.model.weights, next_features))
    td_error = float(np.clip(reward + gamma * next_q - current_q, -TD_CLIP, TD_CLIP))
    self.eligibility *= gamma * lambda_
    self.eligibility += features
    self.model.weights *= 1.0 - alpha * L2
    self.model.weights += alpha * td_error * self.eligibility
    self.model.updates += 1
    self.last_td_error = td_error
    self.episode_td_errors.append(abs(td_error))


def complete_pending_update(self, next_state, next_action):
    if self.pending_transition is None:
        return
    features, reward = self.pending_transition
    next_matrix = self.feature_extractor.action_features(next_state)
    _update(self, features, reward, next_matrix[ACTIONS.index(next_action)])
    self.pending_transition = None


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    features = self.feature_extractor.action_features(old_game_state)[ACTIONS.index(self_action)].copy()
    reward = _transition_reward(self, old_game_state, self_action, new_game_state, events)
    if self.pending_transition is not None:
        raise RuntimeError("SARSA pending transition was not consumed by act()")
    self.pending_transition = (features, reward)


def end_of_round(self, last_game_state, last_action, events):
    # The last action has no following act callback, so update it as terminal.
    features = self.feature_extractor.action_features(last_game_state)[ACTIONS.index(last_action)].copy()
    reward = _transition_reward(self, last_game_state, last_action, None, events)
    _update(self, features, reward, next_features=None)
    self.pending_transition = None
    self.model.episodes += 1
    self.model.save(MODEL_FILE)
    self.logger.info(
        "episode=%d epsilon=%.4f |w|=%.3f mean|td|=%.3f",
        self.model.episodes,
        exploration_rate(self),
        float(np.linalg.norm(self.model.weights)),
        float(np.mean(self.episode_td_errors)) if self.episode_td_errors else 0.0,
    )
    self.eligibility.fill(0.0)
    self.episode_td_errors.clear()
