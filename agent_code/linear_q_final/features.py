"""Task-4 interaction features layered on the established linear baseline."""

from __future__ import annotations

import numpy as np

from agent_code.linear_q_agent.features import (
    ACTIONS, FEATURE_NAMES as BASE_FEATURE_NAMES, MOVE_DELTAS,
    FeatureExtractor as BaseFeatureExtractor,
)

INTERACTION_NAMES = (
    "safe_crate_bomb",
    "safe_opponent_bomb",
    "narrow_bomb_escape",
    "danger_low_freedom",
    "safe_opponent_progress",
    "enemy_adjacent",
    "enemy_nearby",
    "late_hunt_progress",
)
FEATURE_NAMES = BASE_FEATURE_NAMES + INTERACTION_NAMES


class FeatureExtractor:
    def __init__(self, *args, **kwargs):
        self.base = BaseFeatureExtractor(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def feature_count(self):
        return len(FEATURE_NAMES)

    def action_features(self, game_state):
        base = self.base.action_features(game_state)
        extra = np.zeros((len(ACTIONS), len(INTERACTION_NAMES)), dtype=np.float64)
        current = tuple(game_state["self"][-1])
        opponents = [tuple(other[-1]) for other in game_state["others"]]
        hunt_phase = not game_state["coins"] and not np.any(game_state["field"] == 1)
        for index, action in enumerate(ACTIONS):
            target = (current[0] + MOVE_DELTAS[action][0], current[1] + MOVE_DELTAS[action][1])
            nearest = min((abs(target[0] - x) + abs(target[1] - y) for x, y in opponents), default=99)
            escape, danger = base[index, 5], base[index, 6]
            opp_progress = base[index, 10]
            bomb_escape = base[index, 13]
            freedom = base[index, 14]
            extra[index] = (
                bomb_escape * base[index, 11],
                bomb_escape * base[index, 12],
                float(action == "BOMB") * bomb_escape * (1.0 - escape),
                danger * (1.0 - freedom),
                opp_progress * (1.0 - danger) * escape,
                float(nearest <= 1),
                float(nearest <= 3) / max(nearest, 1),
                float(hunt_phase) * opp_progress * min(game_state.get("step", 1) / 400.0, 1.0),
            )
        return np.concatenate((base, extra), axis=1)
