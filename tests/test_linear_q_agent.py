import unittest

import numpy as np

from agent_code.linear_q_agent.features import ACTIONS, FeatureExtractor
from agent_code.linear_q_agent.model import LinearQModel
from agent_code.linear_q_agent.symmetry import D4_TRANSFORMS, transform_action, transform_state
from agent_code.linear_q_agent.train import GAMMA, _discounted_rewards


def make_state():
    field = np.zeros((9, 9), dtype=int)
    field[0, :] = field[-1, :] = -1
    field[:, 0] = field[:, -1] = -1
    field[2, 2] = field[4, 4] = field[6, 6] = -1
    field[5, 3] = 1
    return {
        "round": 1,
        "step": 7,
        "field": field,
        "bombs": [((2, 5), 2)],
        "explosion_map": np.zeros_like(field),
        "coins": [(6, 2)],
        "self": ("learner", 0, True, (3, 5)),
        "others": [("opponent", 0, True, (6, 5))],
        "user_input": None,
    }


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor(horizon=5)

    def test_feature_shape_and_finiteness(self):
        features = self.extractor.action_features(make_state())
        self.assertEqual(features.shape, (len(ACTIONS), self.extractor.feature_count))
        self.assertTrue(np.isfinite(features).all())

    def test_illegal_move_and_bomb_availability(self):
        state = make_state()
        state["field"][3, 4] = -1
        state["self"] = ("learner", 0, False, (3, 5))
        legal = self.extractor.legal_action_mask(state)
        self.assertFalse(legal[ACTIONS.index("UP")])
        self.assertFalse(legal[ACTIONS.index("BOMB")])
        self.assertTrue(legal[ACTIONS.index("WAIT")])

    def test_blast_stops_at_stone_wall(self):
        state = make_state()
        field = state["field"]
        field[4, 5] = -1
        blast = self.extractor.blast_tiles(field, (3, 5))
        self.assertNotIn((4, 5), blast)
        self.assertNotIn((5, 5), blast)
        self.assertIn((3, 2), blast)

    def test_exact_d4_equivariance(self):
        state = make_state()
        original = self.extractor.action_features(state)
        for transform in D4_TRANSFORMS:
            transformed_state = transform_state(state, transform)
            transformed = self.extractor.action_features(transformed_state)
            for action_index, action in enumerate(ACTIONS):
                mapped_index = ACTIONS.index(transform_action(action, transform))
                np.testing.assert_allclose(original[action_index], transformed[mapped_index], atol=1e-12)

    def test_model_update_moves_prediction_towards_target(self):
        model = LinearQModel(self.extractor.feature_count)
        features = self.extractor.action_features(make_state())[ACTIONS.index("RIGHT")]
        before = float(np.dot(features, model.weights))
        model.update(features, target=2.0, alpha=0.01)
        after = float(np.dot(features, model.weights))
        self.assertGreater(after, before)

    def test_n_step_discounted_reward(self):
        buffer = [(None, 1.0), (None, 2.0), (None, 3.0)]
        expected = 1.0 + GAMMA * 2.0 + (GAMMA ** 2) * 3.0
        self.assertAlmostEqual(_discounted_rewards(buffer, 3), expected)


if __name__ == "__main__":
    unittest.main()
