import unittest
import numpy as np
import torch
import test_engine_aligned as engine_tests

from agent_code.dueling_engine_aligned.features import FeatureExtractor as Previous
from agent_code.dueling_priority23.features import FeatureExtractor, ACTIONS
from agent_code.dueling_priority23.callbacks import migrate_policy_input
from agent_code.dueling_priority23.model import DuelingQNetwork
from agent_code.dueling_priority23.symmetry import D4_TRANSFORMS, transform_state, transform_action_features


def state():
    field = np.zeros((17, 17), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return {"field": field, "explosion_map": np.zeros_like(field), "round": 1, "step": 20,
            "self": ("candidate", 0, True, (5, 5)), "others": [], "bombs": [], "coins": []}


class Priority23Tests(unittest.TestCase):
    def test_disabled_flags_reproduce_previous_features_and_mask(self):
        s = state()
        s["field"][7, 5] = 1
        s["coins"] = [(3, 5)]
        s["bombs"] = [((7, 3), 2)]
        old, new = Previous(), FeatureExtractor(enable_economy=False, enable_contest=False)
        np.testing.assert_allclose(new.action_features(s)[:, :22], old.action_features(s))
        np.testing.assert_equal(new.action_features(s)[:, 22:], 0)
        np.testing.assert_equal(new.safe_action_mask(s), old.safe_action_mask(s))

    def test_only_already_covered_crates_are_removed(self):
        s = state()
        s["field"][7, 5] = s["field"][3, 5] = 1
        s["bombs"] = [((7, 3), 2)]
        fx = FeatureExtractor()
        self.assertEqual(fx.bomb_crate_tiles(s, (5, 5)), {(3, 5)})
        f = fx.action_features(s)
        self.assertAlmostEqual(f[5, 11], 1 / 12)
        self.assertEqual(f[5, 22], 0.5)
        # Cache must also invalidate when bombs change without a field change.
        s["bombs"] = []
        self.assertEqual(len(fx.bomb_crate_tiles(s, (5, 5))), 2)

    def test_fully_redundant_position_loses_productive_target(self):
        s = state()
        s["field"][7, 5] = 1
        s["bombs"] = [((7, 3), 2)]
        fx = FeatureExtractor()
        self.assertNotIn((5, 5), fx.productive_bomb_targets(s))
        self.assertEqual(fx.action_features(s)[5, 20], 0)

    def test_no_invented_hidden_coin_targets(self):
        s = state()
        s["field"][7, 5] = 1
        s["bombs"] = [((5, 3), 0)]
        fx = FeatureExtractor()
        np.testing.assert_equal(fx.action_features(s)[:, 23], 0)
        s["coins"] = [(1, 3)]
        self.assertGreater(fx.post_escape_access(s, "LEFT"), fx.post_escape_access(s, "RIGHT"))

    def test_fatal_collision_rejected_when_reliable_escape_exists(self):
        s = state()
        s["bombs"] = [((5, 3), 0)]
        s["others"] = [("enemy", 0, True, (7, 5))]
        self.assertTrue(Previous().safe_action_mask(s)[1])
        mask = FeatureExtractor().safe_action_mask(s)
        self.assertFalse(mask[1])
        self.assertTrue(mask[3])

    def test_last_chance_escape_is_not_discarded(self):
        s = state()
        s["field"][:] = -1
        for p in ((5, 5), (6, 5), (7, 5), (5, 4)):
            s["field"][p] = 0
        s["bombs"] = [((5, 4), 0)]
        s["others"] = [("enemy", 0, True, (7, 5))]
        self.assertTrue(FeatureExtractor().safe_action_mask(s)[1])

    def test_nearby_contested_choke_but_not_all_single_corridors(self):
        s = state()
        s["self"] = ("candidate", 0, True, (3, 3))
        s["field"][:] = -1
        for p in ((3, 3), (4, 3), (5, 3), (5, 4), (5, 5), (6, 5), (13, 13)):
            s["field"][p] = 0
        s["field"][2, 3] = 1
        fx = FeatureExtractor()
        self.assertTrue(fx.safe_action_mask(s)[5])
        s["others"] = [("enemy", 0, True, (13, 13))]
        self.assertTrue(fx.safe_action_mask(s)[5])
        s["others"] = [("enemy", 0, True, (6, 5))]
        self.assertTrue(Previous().safe_action_mask(s)[5])
        self.assertFalse(fx.safe_action_mask(s)[5])

    def test_new_features_respect_all_d4_transforms(self):
        s = state()
        s["field"][7, 5] = 1
        s["bombs"] = [((5, 3), 1)]
        s["coins"] = [(2, 5)]
        s["others"] = [("enemy", 0, True, (7, 6))]
        original = FeatureExtractor().action_features(s)
        for transform in D4_TRANSFORMS:
            actual = FeatureExtractor().action_features(transform_state(s, transform))
            expected = transform_action_features(original.reshape(-1), transform, 26).reshape(6, 26)
            np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_migration_preserves_q_values_for_old_feature_prefix(self):
        old, new = DuelingQNetwork(6 * 22, 6), DuelingQNetwork(6 * 26, 6)
        weights, migrated = migrate_policy_input(old.state_dict(), new.state_dict())
        self.assertTrue(migrated)
        new.load_state_dict(weights)
        rows = torch.randn(1, 6, 26)
        torch.testing.assert_close(old(rows[:, :, :22].reshape(1, -1)), new(rows.reshape(1, -1)))


class Priority23EngineTimingTests(engine_tests.EngineTimingTests):
    """The new subclass must retain every previously verified engine invariant."""
    def setUp(self):
        super().setUp()
        self.fx = FeatureExtractor()


if __name__ == "__main__":
    unittest.main()
