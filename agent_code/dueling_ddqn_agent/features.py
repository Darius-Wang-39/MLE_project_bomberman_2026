"""Action-relative, teacher-compliant features for the linear Q agent."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
MOVE_DELTAS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "WAIT": (0, 0),
    "BOMB": (0, 0),
}
CARDINAL_DELTAS = tuple(MOVE_DELTAS[action] for action in ACTIONS[:4])
HUNT_CENTER_MIN = 3
HUNT_CENTER_MAX = 13
HUNT_CENTER_WEIGHT = 0.20

FEATURE_NAMES = (
    "bias",
    "is_move",
    "is_wait",
    "is_bomb",
    "is_legal",
    "escape_space",
    "target_danger",
    "danger_reduction",
    "coin_progress",
    "crate_progress",
    "opponent_progress",
    "bomb_crate_yield",
    "bomb_opponent_yield",
    "bomb_escape",
    "local_freedom",
    "wait_in_danger",
    "late_round_bomb",
    "hunt_center_progress",
)


def _add(position: Tuple[int, int], delta: Tuple[int, int]) -> Tuple[int, int]:
    return position[0] + delta[0], position[1] + delta[1]


class FeatureExtractor:
    """Compute one feature vector per action.

    Directional actions share the same feature weights. No feature identifies an
    absolute direction or directly returns a hand-written best action.
    """

    def __init__(
        self,
        bomb_power: int = 3,
        bomb_timer: int = 4,
        explosion_duration: int = 2,
        horizon: int = 6,
        hunt_center_min: int = HUNT_CENTER_MIN,
        hunt_center_max: int = HUNT_CENTER_MAX,
        hunt_center_weight: float = HUNT_CENTER_WEIGHT,
    ):
        self.bomb_power = bomb_power
        self.bomb_timer = bomb_timer
        self.explosion_duration = explosion_duration
        self.horizon = horizon
        self.hunt_center_min = hunt_center_min
        self.hunt_center_max = hunt_center_max
        self.hunt_center_weight = hunt_center_weight

    @property
    def feature_count(self) -> int:
        return len(FEATURE_NAMES)

    def blast_tiles(self, field: np.ndarray, origin: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Match the supplied framework: stone walls stop blasts; crates do not."""
        tiles = {origin}
        for dx, dy in CARDINAL_DELTAS:
            for distance in range(1, self.bomb_power + 1):
                tile = origin[0] + dx * distance, origin[1] + dy * distance
                if field[tile] == -1:
                    break
                tiles.add(tile)
        return tiles

    def legal_action_mask(self, game_state: Dict) -> np.ndarray:
        field = game_state["field"]
        position = tuple(game_state["self"][-1])
        bombs = {tuple(bomb_position) for bomb_position, _ in game_state["bombs"]}
        others = {tuple(other[-1]) for other in game_state["others"]}
        mask = np.ones(len(ACTIONS), dtype=bool)
        for index, action in enumerate(ACTIONS[:4]):
            target = _add(position, MOVE_DELTAS[action])
            mask[index] = field[target] == 0 and target not in bombs and target not in others
        mask[ACTIONS.index("WAIT")] = True
        mask[ACTIONS.index("BOMB")] = bool(game_state["self"][2]) and position not in bombs
        return mask

    def _bomb_schedule(self, game_state: Dict, add_own_bomb: bool) -> List[Tuple[Set[Tuple[int, int]], int, Tuple[int, int]]]:
        field = game_state["field"]
        schedule = []
        for position, timer in game_state["bombs"]:
            position = tuple(position)
            schedule.append((self.blast_tiles(field, position), int(timer) + 1, position))
        if add_own_bomb:
            position = tuple(game_state["self"][-1])
            schedule.append((self.blast_tiles(field, position), self.bomb_timer + 1, position))
        return schedule

    def _is_dangerous(
        self,
        position: Tuple[int, int],
        time_step: int,
        schedule: Sequence[Tuple[Set[Tuple[int, int]], int, Tuple[int, int]]],
        explosion_map: np.ndarray,
    ) -> bool:
        if explosion_map[position] > 0 and time_step <= int(explosion_map[position]) + 1:
            return True
        return any(
            position in blast and explosion_time <= time_step < explosion_time + self.explosion_duration
            for blast, explosion_time, _ in schedule
        )

    def _bomb_blocks(
        self,
        position: Tuple[int, int],
        time_step: int,
        schedule: Sequence[Tuple[Set[Tuple[int, int]], int, Tuple[int, int]]],
    ) -> bool:
        return any(position == bomb_position and time_step < explosion_time for _, explosion_time, bomb_position in schedule)

    def survival_count(self, game_state: Dict, action: str) -> int:
        """Count distinct positions reachable at the planning horizon after an action."""
        legal = self.legal_action_mask(game_state)
        action_index = ACTIONS.index(action)
        if not legal[action_index]:
            return 0

        field = game_state["field"]
        explosion_map = game_state["explosion_map"]
        start = tuple(game_state["self"][-1])
        first_position = _add(start, MOVE_DELTAS[action])
        schedule = self._bomb_schedule(game_state, add_own_bomb=action == "BOMB")

        if self._is_dangerous(first_position, 1, schedule, explosion_map):
            return 0

        positions = {first_position}
        for time_step in range(2, self.horizon + 1):
            next_positions: Set[Tuple[int, int]] = set()
            for position in positions:
                for delta in CARDINAL_DELTAS + ((0, 0),):
                    target = _add(position, delta)
                    if field[target] != 0:
                        continue
                    if self._bomb_blocks(target, time_step, schedule):
                        continue
                    if self._is_dangerous(target, time_step, schedule, explosion_map):
                        continue
                    next_positions.add(target)
            positions = next_positions
            if not positions:
                return 0
        return len(positions)

    def safe_action_mask(self, game_state: Dict) -> np.ndarray:
        legal = self.legal_action_mask(game_state)
        survival = np.array(
            [self.survival_count(game_state, action) if legal[index] else 0 for index, action in enumerate(ACTIONS)],
            dtype=int,
        )
        safe = legal & (survival > 0)
        return safe if safe.any() else legal

    def _distance_map(
        self,
        field: np.ndarray,
        targets: Iterable[Tuple[int, int]],
        blocked: Set[Tuple[int, int]],
    ) -> np.ndarray:
        distances = np.full(field.shape, np.inf, dtype=float)
        queue = deque()
        for target in set(targets):
            if field[target] == 0:
                distances[target] = 0.0
                queue.append(target)
        while queue:
            position = queue.popleft()
            for delta in CARDINAL_DELTAS:
                neighbor = _add(position, delta)
                if field[neighbor] != 0 or neighbor in blocked:
                    continue
                candidate = distances[position] + 1.0
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    queue.append(neighbor)
        return distances

    def _target_maps(self, game_state: Dict):
        field = game_state["field"]
        bomb_positions = {tuple(position) for position, _ in game_state["bombs"]}
        opponent_positions = {tuple(other[-1]) for other in game_state["others"]}
        blocked = bomb_positions | opponent_positions

        coin_map = self._distance_map(field, map(tuple, game_state["coins"]), blocked)

        crate_targets = set()
        for x, y in np.argwhere(field == 1):
            for delta in CARDINAL_DELTAS:
                neighbor = _add((int(x), int(y)), delta)
                if field[neighbor] == 0:
                    crate_targets.add(neighbor)
        crate_map = self._distance_map(field, crate_targets, blocked)

        opponent_targets = set()
        for opponent in opponent_positions:
            for delta in CARDINAL_DELTAS:
                neighbor = _add(opponent, delta)
                if field[neighbor] == 0:
                    opponent_targets.add(neighbor)
        opponent_map = self._distance_map(field, opponent_targets, bomb_positions)
        return coin_map, crate_map, opponent_map

    @staticmethod
    def _progress(distance_map: np.ndarray, current: Tuple[int, int], target: Tuple[int, int]) -> float:
        before = distance_map[current]
        after = distance_map[target]
        if not np.isfinite(before) or not np.isfinite(after):
            return 0.0
        return float(np.clip(before - after, -1.0, 1.0))

    def _danger_score(self, game_state: Dict, position: Tuple[int, int]) -> float:
        if game_state["explosion_map"][position] > 0:
            return 1.0
        schedule = self._bomb_schedule(game_state, add_own_bomb=False)
        times = [explosion_time for blast, explosion_time, _ in schedule if position in blast]
        return 0.0 if not times else 1.0 / float(1 + min(times))

    def _local_freedom(self, game_state: Dict, position: Tuple[int, int]) -> float:
        field = game_state["field"]
        bombs = {tuple(bomb_position) for bomb_position, _ in game_state["bombs"]}
        others = {tuple(other[-1]) for other in game_state["others"]}
        free = sum(
            field[_add(position, delta)] == 0
            and _add(position, delta) not in bombs
            and _add(position, delta) not in others
            for delta in CARDINAL_DELTAS
        )
        return free / 4.0

    def _hunt_phase(self, game_state: Dict) -> bool:
        return (
            not game_state["coins"]
            and not np.any(game_state["field"] == 1)
            and bool(game_state["others"])
        )

    def center_box_distance(self, position: Tuple[int, int]) -> int:
        """Manhattan distance to the inclusive central 11x11 box."""
        x, y = position
        dx = max(self.hunt_center_min - x, 0, x - self.hunt_center_max)
        dy = max(self.hunt_center_min - y, 0, y - self.hunt_center_max)
        return int(dx + dy)

    def action_features(self, game_state: Dict) -> np.ndarray:
        legal = self.legal_action_mask(game_state)
        survival = np.array(
            [self.survival_count(game_state, action) if legal[index] else 0 for index, action in enumerate(ACTIONS)],
            dtype=float,
        )
        coin_map, crate_map, opponent_map = self._target_maps(game_state)
        field = game_state["field"]
        current = tuple(game_state["self"][-1])
        current_danger = self._danger_score(game_state, current)
        late_round = min(float(game_state.get("step", 1)) / 400.0, 1.0)
        hunt_phase = self._hunt_phase(game_state)
        current_center_distance = self.center_box_distance(current)
        features = np.zeros((len(ACTIONS), self.feature_count), dtype=np.float64)

        blast = self.blast_tiles(field, current)
        crates_in_blast = sum(field[tile] == 1 for tile in blast) / 12.0
        opponents = {tuple(other[-1]) for other in game_state["others"]}
        opponents_in_blast = sum(tile in opponents for tile in blast) / 3.0

        for index, action in enumerate(ACTIONS):
            target = _add(current, MOVE_DELTAS[action])
            target_danger = self._danger_score(game_state, target) if legal[index] else 1.0
            row = features[index]
            row[0] = 1.0
            row[1] = float(action in ACTIONS[:4])
            row[2] = float(action == "WAIT")
            row[3] = float(action == "BOMB")
            row[4] = float(legal[index])
            row[5] = min(survival[index] / 20.0, 1.0)
            row[6] = target_danger
            row[7] = np.clip(current_danger - target_danger, -1.0, 1.0)
            row[8] = self._progress(coin_map, current, target) if legal[index] else 0.0
            row[9] = self._progress(crate_map, current, target) if legal[index] else 0.0
            row[10] = self._progress(opponent_map, current, target) if legal[index] else 0.0
            row[11] = crates_in_blast if action == "BOMB" else 0.0
            row[12] = opponents_in_blast if action == "BOMB" else 0.0
            row[13] = float(action == "BOMB" and survival[index] > 0)
            row[14] = self._local_freedom(game_state, target) if legal[index] else 0.0
            row[15] = float(action == "WAIT") * current_danger
            row[16] = float(action == "BOMB") * late_round
            row[17] = (
                float(current_center_distance - self.center_box_distance(target))
                if legal[index] and hunt_phase
                else 0.0
            )
        return features

    def state_potential(self, game_state: Dict | None) -> float:
        """State-only potential used for policy-invariant dense shaping."""
        if game_state is None:
            return 0.0
        coin_map, crate_map, opponent_map = self._target_maps(game_state)
        position = tuple(game_state["self"][-1])

        # Use a positive closeness potential so merely waiting at a target does
        # not receive a positive shaping reward under gamma < 1. Coins remain
        # the first objective, then crates, and opponents become the explicit
        # objective only in the otherwise target-free endgame.
        hunt_phase = self._hunt_phase(game_state)
        if game_state["coins"]:
            target_distance = coin_map[position]
            distance_scale = 20.0
            target_weight = 1.0
        elif np.any(game_state["field"] == 1):
            target_distance = crate_map[position]
            distance_scale = 20.0
            target_weight = 1.0
        elif hunt_phase:
            target_distance = opponent_map[position]
            distance_scale = 30.0
            target_weight = 2.0
        else:
            target_distance = np.inf
            distance_scale = 1.0
            target_weight = 0.0

        distance_term = 0.0
        if np.isfinite(target_distance):
            closeness = 1.0 - min(float(target_distance), distance_scale) / distance_scale
            distance_term = target_weight * closeness
        center_term = 0.0
        if hunt_phase:
            playable_max_x = game_state["field"].shape[0] - 2
            playable_max_y = game_state["field"].shape[1] - 2
            max_dx = max(self.hunt_center_min - 1, playable_max_x - self.hunt_center_max)
            max_dy = max(self.hunt_center_min - 1, playable_max_y - self.hunt_center_max)
            max_center_distance = max(max_dx + max_dy, 1)
            center_closeness = 1.0 - min(
                self.center_box_distance(position), max_center_distance
            ) / float(max_center_distance)
            center_term = self.hunt_center_weight * center_closeness
        danger_term = -self._danger_score(game_state, position)
        return distance_term + center_term + danger_term
