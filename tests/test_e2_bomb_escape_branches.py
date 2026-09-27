import unittest

import numpy as np

from evaluation.e2_bomb_escape_branches import apply_branch_rule


class BombBranchRuleTests(unittest.TestCase):
    def test_only_bomb_permission_changes(self):
        allowed = np.array([True, False, True, True, True, True])
        strict = apply_branch_rule(allowed, branch_count=1, minimum_branches=2)
        relaxed = apply_branch_rule(allowed, branch_count=1, minimum_branches=1)
        np.testing.assert_array_equal(strict[:-1], relaxed[:-1])
        self.assertFalse(strict[-1])
        self.assertTrue(relaxed[-1])

    def test_both_arms_accept_two_independent_branches(self):
        allowed = np.ones(6, dtype=bool)
        for threshold in (1, 2):
            self.assertTrue(apply_branch_rule(allowed, 2, threshold)[-1])

    def test_neither_arm_revives_passively_lethal_bomb(self):
        allowed = np.ones(6, dtype=bool)
        allowed[-1] = False
        for threshold in (1, 2):
            self.assertFalse(apply_branch_rule(allowed, 3, threshold)[-1])


if __name__ == "__main__":
    unittest.main()
