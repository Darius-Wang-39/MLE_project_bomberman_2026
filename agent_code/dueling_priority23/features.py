"""P2 resource efficiency and P3 contested exits, on the corrected engine model.

2026-09-16: independent experimental agent; historical policies are untouched.
Opponent arrival is a bounded risk estimate, not a prediction of their actions.
Only already visible coins are used; hidden coin locations are never available.
"""
from __future__ import annotations

import numpy as np

from . import base_features as base
from .base_features import ACTIONS, MOVE_DELTAS, CARDINAL_DELTAS, _add

FEATURE_NAMES = base.FEATURE_NAMES + (
    "redundant_bomb_fraction", "post_escape_coin_access",
    "contested_move", "bomb_exit_robustness",
)


class FeatureExtractor(base.FeatureExtractor):
    def __init__(self, *args, enable_economy=True, enable_contest=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_economy = enable_economy
        self.enable_contest = enable_contest
        self._risk_key = None
        self._risk_cache = {}

    @property
    def feature_count(self):
        return len(FEATURE_NAMES)

    @property
    def feature_names(self):
        return FEATURE_NAMES

    def _cache(self, state):
        key = (state["field"].tobytes(), state["explosion_map"].tobytes(),
               tuple((tuple(p), int(t)) for p, t in state["bombs"]),
               tuple((a[0], a[2], tuple(a[-1])) for a in state["others"]),
               tuple(state["self"]), tuple(map(tuple, state["coins"])))
        if key != self._risk_key:
            self._risk_key, self._risk_cache = key, {}
        return self._risk_cache

    def scheduled_crates(self, state):
        cache = self._cache(state)
        if "scheduled" not in cache:
            cache["scheduled"] = {
                tile for blast, step, _ in self._bomb_schedule(state, False)
                if step <= self.bomb_timer + 1 for tile in blast
                if state["field"][tile] == 1
            }
        return cache["scheduled"]

    def bomb_crate_tiles(self, state, position):
        tiles = super().bomb_crate_tiles(state, position)
        # PRIORITY-2: crates already doomed before our new bomb are not new value.
        return tiles - self.scheduled_crates(state) if self.enable_economy else tiles

    def productive_bomb_targets(self, state):
        targets = super().productive_bomb_targets(state)
        if not self.enable_economy or not state["bombs"]:
            return targets
        return {p for p in targets if self.bomb_crate_tiles(state, p)}

    def _target_maps(self, state):
        coin_map, crate_map, opponent_map = super()._target_maps(state)
        if self.enable_economy:
            blocked = {tuple(p) for p, _ in state["bombs"]}
            blocked.update(tuple(a[-1]) for a in state["others"])
            crate_map = self._distance_map(state["field"], self.productive_bomb_targets(state), blocked)
        return coin_map, crate_map, opponent_map

    def opponent_arrivals(self, state):
        cache = self._cache(state)
        if "arrivals" not in cache:
            blocked = {tuple(p) for p, _ in state["bombs"]}
            cache["arrivals"] = self._distance_map(
                state["field"], [tuple(a[-1]) for a in state["others"]], blocked)
        return cache["arrivals"]

    def contested_move(self, state, action):
        if action not in ACTIONS[:4] or not state["others"]:
            return False
        target = _add(tuple(state["self"][-1]), MOVE_DELTAS[action])
        return bool(self.opponent_arrivals(state)[target] <= 1)

    def future_endpoints(self, state, action, contest=False):
        cache = self._cache(state)
        key = ("endpoints", action, contest)
        if key in cache:
            return cache[key]
        field = state["field"]
        if not self.legal_action_mask(state)[ACTIONS.index(action)]:
            return set()
        schedule = self._bomb_schedule(state, action == "BOMB")
        start = tuple(state["self"][-1])
        first = _add(start, MOVE_DELTAS[action])
        if self._is_dangerous(first, 1, schedule, state["explosion_map"]):
            return set()
        positions = {first}
        opponents = {tuple(a[-1]) for a in state["others"]}
        arrivals = self.opponent_arrivals(state) if contest else None
        # PRIORITY-3: stress-test nearby narrow tiles, not every single corridor.
        # A distant opponent, or a wide open tile, does not trigger this filter.
        for step in range(2, self.horizon + 1):
            following = set()
            for position in positions:
                for delta in CARDINAL_DELTAS + ((0, 0),):
                    target = _add(position, delta)
                    if field[target] != 0 or target in opponents:
                        continue
                    if self._bomb_blocks(target, step, schedule):
                        continue
                    if self._is_dangerous(target, step, schedule, state["explosion_map"]):
                        continue
                    if contest and arrivals[target] <= min(step, 3):
                        degree = sum(field[_add(target, d)] == 0 for d in CARDINAL_DELTAS)
                        if degree <= 2:
                            continue
                    following.add(target)
            positions = following
            if not positions:
                break
        cache[key] = positions
        return positions

    def bomb_robustness(self, state):
        normal = self.future_endpoints(state, "BOMB")
        if not normal:
            return 0.0
        robust = self.future_endpoints(state, "BOMB", contest=True)
        return len(robust) / len(normal)

    def safe_action_mask(self, state):
        safe = super().safe_action_mask(state)
        if not self.enable_contest or not state["others"]:
            return safe
        revised = safe.copy()
        schedule = self._bomb_schedule(state, False)
        current = tuple(state["self"][-1])
        collision_is_fatal = self._is_dangerous(current, 1, schedule, state["explosion_map"])
        for index, action in enumerate(ACTIONS[:4]):
            if revised[index] and collision_is_fatal and self.contested_move(state, action):
                revised[index] = False
        bomb = ACTIONS.index("BOMB")
        if revised[bomb] and self.bomb_robustness(state) == 0.0:
            revised[bomb] = False
        # Do not discard the last possible escape just because collision is possible.
        return revised if revised.any() else safe

    def post_escape_access(self, state, action):
        if not state["coins"] or not state["bombs"]:
            return 0.0
        endpoints = self.future_endpoints(state, action)
        if not endpoints:
            return 0.0
        cache = self._cache(state)
        if "future_coins" not in cache:
            future_field = state["field"].copy()
            # Known crates will be gone by our horizon; do not invent hidden coins.
            for tile in self.scheduled_crates(state):
                future_field[tile] = 0
            cache["future_coins"] = self._distance_map(future_field, map(tuple, state["coins"]), set())
        distance = min(cache["future_coins"][p] for p in endpoints)
        return 1.0 / (1.0 + distance) if np.isfinite(distance) else 0.0

    def action_features(self, state):
        features = super().action_features(state)
        current = tuple(state["self"][-1])
        raw = super().bomb_crate_tiles(state, current)
        bomb = ACTIONS.index("BOMB")
        if self.enable_economy:
            if raw:
                features[bomb, 22] = len(raw & self.scheduled_crates(state)) / len(raw)
            for index, action in enumerate(ACTIONS):
                features[index, 23] = self.post_escape_access(state, action)
            # PRIORITY-2: when escape quality ties, prefer a safe endpoint closer
            # to visible coins. Safety remains primary; there is no extra Q bias.
            safe = self.safe_action_mask(state)
            if self._danger_score(state, current) > 0 and state["coins"] and safe.any():
                score = features[:, 7] + features[:, 5]
                best = safe & np.isclose(score, np.max(score[safe]))
                features[:, 21] = best & np.isclose(features[:, 23], np.max(features[best, 23]))
        if self.enable_contest:
            for index, action in enumerate(ACTIONS):
                features[index, 24] = float(self.contested_move(state, action))
            features[bomb, 25] = self.bomb_robustness(state)
        return features
