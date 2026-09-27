import unittest

import numpy as np

from evaluation.e4_opponent_routing_gate import (
    hunt_phase,
    opponent_route_enabled,
)


def sample_state():
    field = -np.ones((7, 7), dtype=int)
    field[1:6, 1:6] = 0
    return {
        "round": 1, "step": 1, "field": field,
        "self": ("candidate", 0, True, (3, 3)),
        "others": [("other", 0, True, (5, 5))],
        "bombs": [], "coins": [], "explosion_map": np.zeros_like(field),
    }


class OpponentRoutingGateTests(unittest.TestCase):
    def test_non_opponent_objectives_are_never_gated(self):
        state = sample_state()
        for variant in (
            "opponent_route_off", "fallback_final", "strict_hunt_only"
        ):
            for source in (
                "escape", "coin", "immediate_crate_bomb",
                "productive_crate_target", "none",
            ):
                self.assertTrue(opponent_route_enabled(variant, source, state))

    def test_three_prespecified_opponent_gates(self):
        state = sample_state()
        self.assertTrue(hunt_phase(state))
        self.assertFalse(opponent_route_enabled("opponent_route_off", "opponent", state))
        self.assertTrue(opponent_route_enabled("fallback_final", "opponent", state))
        self.assertTrue(opponent_route_enabled("strict_hunt_only", "opponent", state))

        state["coins"] = [(1, 1)]
        self.assertFalse(hunt_phase(state))
        self.assertTrue(opponent_route_enabled("fallback_final", "opponent", state))
        self.assertFalse(opponent_route_enabled("strict_hunt_only", "opponent", state))

    def test_crates_prevent_strict_hunt(self):
        state = sample_state()
        state["field"][2, 2] = 1
        self.assertFalse(hunt_phase(state))
        self.assertFalse(opponent_route_enabled("strict_hunt_only", "opponent", state))


if __name__ == "__main__":
    unittest.main()
