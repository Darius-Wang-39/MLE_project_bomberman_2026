"""One reward function shared by every controlled learner."""

from __future__ import annotations

from typing import List

import events as e


GAMMA = 0.95
SHAPING_SCALE = 0.5
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


def transition_reward(self, old_state, action: str, new_state, events: List[str]) -> float:
    """Return an action-independent event reward plus shared potential shaping."""
    del action
    event_reward = float(sum(EVENT_REWARDS.get(event, 0.0) for event in events))
    shaping = SHAPING_SCALE * (
        GAMMA * self.feature_extractor.state_potential(new_state)
        - self.feature_extractor.state_potential(old_state)
    )
    return event_reward + shaping
