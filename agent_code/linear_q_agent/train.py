"""Online Q-learning updates and potential-based reward shaping."""

from __future__ import annotations

from collections import deque
from typing import List

import numpy as np

import events as e
from .callbacks import ACTIONS, MODEL_FILE


GAMMA = 0.95
ALPHA = 0.01
SHAPING_SCALE = 0.5
N_STEP = 5

EVENT_REWARDS = {
    e.MOVED_LEFT: -0.02,
    e.MOVED_RIGHT: -0.02,
    e.MOVED_UP: -0.02,
    e.MOVED_DOWN: -0.02,
    e.WAITED: -0.05,
    e.INVALID_ACTION: -2.0,
    e.BOMB_DROPPED: -0.02,
    e.CRATE_DESTROYED: 0.8,
    e.COIN_FOUND: 0.5,
    e.COIN_COLLECTED: 4.0,
    e.KILLED_OPPONENT: 10.0,
    e.KILLED_SELF: -20.0,
    e.GOT_KILLED: -15.0,
    e.SURVIVED_ROUND: 2.0,
}


def setup_training(self):
    self.last_td_error = 0.0
    self.n_step_buffer = deque()


def _transition_reward(self, old_game_state, new_game_state, events: List[str]) -> float:
    reward = reward_from_events(self, events)
    return reward + SHAPING_SCALE * (
        GAMMA * self.feature_extractor.state_potential(new_game_state)
        - self.feature_extractor.state_potential(old_game_state)
    )


def _discounted_rewards(buffer, count: int) -> float:
    return float(sum((GAMMA ** index) * buffer[index][1] for index in range(count)))


def _update_oldest(self, bootstrap_state=None):
    count = min(N_STEP, len(self.n_step_buffer))
    target = _discounted_rewards(self.n_step_buffer, count)
    if bootstrap_state is not None and count == N_STEP:
        next_features = self.feature_extractor.action_features(bootstrap_state)
        next_allowed = self.feature_extractor.safe_action_mask(bootstrap_state)
        next_values = self.model.q_values(next_features)
        next_best = float(np.max(next_values[next_allowed])) if next_allowed.any() else 0.0
        target += (GAMMA ** count) * next_best

    features, _ = self.n_step_buffer.popleft()
    self.last_td_error = self.model.update(features, target, alpha=ALPHA)
    self.logger.debug(
        "linear update=%d n=%d target=%.3f td=%.3f",
        self.model.updates,
        count,
        target,
        self.last_td_error,
    )


def _record_transition(self, old_game_state, action: str, new_game_state, events: List[str]):
    old_features = self.feature_extractor.action_features(old_game_state)
    action_index = ACTIONS.index(action)
    reward = _transition_reward(self, old_game_state, new_game_state, events)
    self.n_step_buffer.append((old_features[action_index].copy(), reward))

    if new_game_state is None:
        while self.n_step_buffer:
            _update_oldest(self, bootstrap_state=None)
    elif len(self.n_step_buffer) >= N_STEP:
        _update_oldest(self, bootstrap_state=new_game_state)


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    _record_transition(self, old_game_state, self_action, new_game_state, events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    _record_transition(self, last_game_state, last_action, None, events)
    self.model.save(MODEL_FILE)


def reward_from_events(self, events: List[str]) -> float:
    reward = float(sum(EVENT_REWARDS.get(event, 0.0) for event in events))
    self.logger.info("Awarded %.3f for events %s", reward, ", ".join(events))
    return reward
