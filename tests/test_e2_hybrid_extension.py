import unittest

import numpy as np

from agent_code.dueling_hunt_center_04.features import FeatureExtractor


def sample_state():
    field = np.zeros((17, 17), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("candidate", 0, True, (2, 2)),
        "others": [("enemy", 0, True, (15, 15))],
        "bombs": [],
        "coins": [],
        "explosion_map": np.zeros_like(field),
    }


class OriginalHybridRuleTests(unittest.TestCase):
    def test_quiet_productive_single_branch_exception_is_preserved(self):
        extractor = FeatureExtractor()
        state = sample_state()
        # UP and LEFT are walls; RIGHT is a crate, leaving DOWN as one branch.
        state["field"][2, 1] = -1
        state["field"][1, 2] = -1
        state["field"][3, 2] = 1
        self.assertEqual(extractor.bomb_escape_branch_count(state), 1)
        self.assertTrue(extractor.bomb_safety_is_acceptable(state))

    def test_nonproductive_single_branch_is_rejected(self):
        extractor = FeatureExtractor()
        state = sample_state()
        state["field"][2, 1] = -1
        state["field"][1, 2] = -1
        state["field"][3, 2] = -1
        self.assertEqual(extractor.bomb_escape_branch_count(state), 1)
        self.assertFalse(extractor.bomb_safety_is_acceptable(state))


if __name__ == "__main__":
    unittest.main()
