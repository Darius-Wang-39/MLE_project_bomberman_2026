import unittest
import numpy as np

from agent_code.linear_q_final.features import ACTIONS, FEATURE_NAMES, FeatureExtractor
from agent_code.linear_q_final.model import LinearQModel
from tests.test_linear_q_agent import make_state


class FinalLinearFeatureTests(unittest.TestCase):
    def test_shape_finite_and_base_prefix(self):
        state = make_state()
        enhanced = FeatureExtractor().action_features(state)
        base = FeatureExtractor().base.action_features(state)
        self.assertEqual(enhanced.shape, (len(ACTIONS), len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(enhanced).all())
        np.testing.assert_allclose(enhanced[:, :base.shape[1]], base)

    def test_old_checkpoint_weights_can_be_padded(self):
        model = LinearQModel(25, weights=np.r_[np.ones(17), np.zeros(8)], updates=1)
        self.assertEqual(model.weights.shape, (25,))


if __name__ == "__main__":
    unittest.main()
