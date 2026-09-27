import unittest

import numpy as np

from agent_code.dueling_hunt_center_04.features import FeatureExtractor
from evaluation.e3_economic_routing import (
    classify_objective_source,
    source_enabled,
)


def sample_state():
    field = -np.ones((7, 7), dtype=int)
    field[1:6, 1:6] = 0
    return {
        "round": 1, "step": 1, "field": field,
        "self": ("candidate", 0, True, (3, 3)),
        "others": [], "bombs": [], "coins": [],
        "explosion_map": np.zeros_like(field),
    }


class EconomicRoutingTests(unittest.TestCase):
    def test_source_switch_preserves_non_economic_routes(self):
        for variant in ("no_economic_route", "coin_only", "full_economic_route"):
            self.assertTrue(source_enabled(variant, "escape"))
            self.assertTrue(source_enabled(variant, "opponent"))
        self.assertFalse(source_enabled("no_economic_route", "coin"))
        self.assertTrue(source_enabled("coin_only", "coin"))
        self.assertFalse(source_enabled("coin_only", "immediate_crate_bomb"))
        self.assertTrue(source_enabled("full_economic_route", "productive_crate_target"))

    def test_classifies_coin_before_crate(self):
        state = sample_state()
        state["coins"] = [(3, 1)]
        state["field"][4, 3] = 1
        extractor = FeatureExtractor()
        features = extractor.action_features(state)
        self.assertEqual(classify_objective_source(extractor, state, features), "coin")

    def test_classifies_immediate_productive_bomb(self):
        state = sample_state()
        state["field"][4, 3] = 1
        extractor = FeatureExtractor()
        features = extractor.action_features(state)
        self.assertEqual(
            classify_objective_source(extractor, state, features),
            "immediate_crate_bomb",
        )


if __name__ == "__main__":
    unittest.main()
