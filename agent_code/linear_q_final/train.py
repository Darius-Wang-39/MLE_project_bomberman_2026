"""Configurable n-step Q-learning for the final controlled experiment."""

from collections import deque
from typing import List
import numpy as np
import events as e
from .callbacks import ACTIONS, MODEL_FILE

GAMMA = 0.95
SHAPING_SCALE = 0.5
EVENT_REWARDS = {
    e.MOVED_LEFT:-0.02, e.MOVED_RIGHT:-0.02, e.MOVED_UP:-0.02, e.MOVED_DOWN:-0.02,
    e.WAITED:-0.05, e.INVALID_ACTION:-2.0, e.BOMB_DROPPED:-0.02,
    e.CRATE_DESTROYED:0.8, e.COIN_FOUND:0.5, e.COIN_COLLECTED:4.0,
    e.KILLED_OPPONENT:10.0, e.KILLED_SELF:-20.0, e.GOT_KILLED:-15.0,
    e.SURVIVED_ROUND:2.0,
}


def setup_training(self):
    self.alpha = float(self.parameters["alpha"])
    self.n_step = int(self.parameters["n_step"])
    self.n_step_buffer = deque()
    self.last_td_error = 0.0


def reward_from_events(self, events):
    return float(sum(EVENT_REWARDS.get(event, 0.0) for event in events))


def _reward(self, old_state, new_state, events):
    return reward_from_events(self, events) + SHAPING_SCALE * (
        GAMMA * self.feature_extractor.state_potential(new_state)
        - self.feature_extractor.state_potential(old_state)
    )


def _discounted(buffer, count):
    return float(sum((GAMMA ** i) * buffer[i][1] for i in range(count)))


def _update_oldest(self, bootstrap_state=None):
    count = min(self.n_step, len(self.n_step_buffer))
    target = _discounted(self.n_step_buffer, count)
    if bootstrap_state is not None and count == self.n_step:
        matrix = self.feature_extractor.action_features(bootstrap_state)
        allowed = self.feature_extractor.safe_action_mask(bootstrap_state)
        values = self.model.q_values(matrix)
        target += (GAMMA ** count) * (float(np.max(values[allowed])) if allowed.any() else 0.0)
    features, _ = self.n_step_buffer.popleft()
    self.last_td_error = self.model.update(features, target, alpha=self.alpha)


def _record(self, old_state, action, new_state, events):
    features = self.feature_extractor.action_features(old_state)[ACTIONS.index(action)].copy()
    self.n_step_buffer.append((features, _reward(self, old_state, new_state, events)))
    if new_state is None:
        while self.n_step_buffer:
            _update_oldest(self)
    elif len(self.n_step_buffer) >= self.n_step:
        _update_oldest(self, new_state)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events: List[str]):
    _record(self, old_game_state, self_action, new_game_state, events)


def end_of_round(self, last_game_state, last_action, events: List[str]):
    _record(self, last_game_state, last_action, None, events)
    self.model.save(MODEL_FILE)
