"""Dihedral symmetries of the square Bomberman board.

The linear agent uses action-relative features, so it shares parameters across
all eight transforms without adding duplicate transitions. These utilities are
also used to test that the feature representation is exactly equivariant.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
DIRECTION_VECTORS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}
VECTOR_DIRECTIONS = {value: key for key, value in DIRECTION_VECTORS.items()}

# Four rotations, followed by the same rotations with a left-right reflection.
D4_TRANSFORMS = tuple((rotation, reflected) for reflected in (False, True) for rotation in range(4))


def transform_vector(vector: Tuple[int, int], transform: Tuple[int, bool]) -> Tuple[int, int]:
    """Transform a direction vector by clockwise rotations and a reflection."""
    x, y = vector
    rotation, reflected = transform
    for _ in range(rotation % 4):
        x, y = -y, x
    if reflected:
        x = -x
    return x, y


def transform_action(action: str, transform: Tuple[int, bool]) -> str:
    if action not in DIRECTION_VECTORS:
        return action
    return VECTOR_DIRECTIONS[transform_vector(DIRECTION_VECTORS[action], transform)]


def transform_coordinate(
    coordinate: Tuple[int, int], board_size: int, transform: Tuple[int, bool]
) -> Tuple[int, int]:
    x, y = coordinate
    rotation, reflected = transform
    for _ in range(rotation % 4):
        x, y = board_size - 1 - y, x
    if reflected:
        x = board_size - 1 - x
    return x, y


def transform_array(array: np.ndarray, transform: Tuple[int, bool]) -> np.ndarray:
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("D4 transforms require a square 2D array")
    size = array.shape[0]
    result = np.empty_like(array)
    for x in range(size):
        for y in range(size):
            tx, ty = transform_coordinate((x, y), size, transform)
            result[tx, ty] = array[x, y]
    return result


def transform_state(game_state: Dict, transform: Tuple[int, bool]) -> Dict:
    """Return a transformed copy of a framework game state."""
    size = game_state["field"].shape[0]

    def coord(value):
        return transform_coordinate(tuple(value), size, transform)

    own_name, own_score, own_bomb, own_pos = game_state["self"]
    transformed = dict(game_state)
    transformed["field"] = transform_array(game_state["field"], transform)
    transformed["explosion_map"] = transform_array(game_state["explosion_map"], transform)
    transformed["coins"] = [coord(position) for position in game_state["coins"]]
    transformed["bombs"] = [(coord(position), timer) for position, timer in game_state["bombs"]]
    transformed["self"] = (own_name, own_score, own_bomb, coord(own_pos))
    transformed["others"] = [
        (name, score, bomb, coord(position))
        for name, score, bomb, position in game_state["others"]
    ]
    if game_state.get("user_input") in ACTIONS:
        transformed["user_input"] = transform_action(game_state["user_input"], transform)
    return transformed

