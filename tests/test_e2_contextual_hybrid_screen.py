import unittest

from evaluation.e2_contextual_hybrid_screen import permissions, strict_trap_flags


def context(**updates):
    value = {
        "bomb_branch_count": 1,
        "crate_yield": 1,
        "crates_remaining": 10,
        "active_bomb_count": 0,
        "active_explosion": False,
        "nearest_opponent_distance": 7,
        "nearby_armed_opponent": False,
        "predicted_branch_threat": False,
    }
    value.update(updates)
    return value


class ContextualHybridPermissionTests(unittest.TestCase):
    def test_phase_rule_removes_only_fixed_step_cutoff(self):
        result = permissions(context(), True, step=200)
        self.assertFalse(result["original_hybrid"])
        self.assertTrue(result["phase_productive_hybrid"])

    def test_phase_rule_is_strict_after_crates_are_gone(self):
        result = permissions(context(crate_yield=0, crates_remaining=0), True, step=50)
        self.assertFalse(result["phase_productive_hybrid"])

    def test_armed_rule_ignores_close_unarmed_opponent(self):
        result = permissions(context(nearest_opponent_distance=2), True, step=50)
        self.assertFalse(result["original_hybrid"])
        self.assertTrue(result["armed_threat_hybrid"])

    def test_armed_rule_rejects_armed_or_predicted_threat(self):
        self.assertFalse(permissions(context(nearby_armed_opponent=True), True, 50)["armed_threat_hybrid"])
        self.assertFalse(permissions(context(predicted_branch_threat=True), True, 50)["armed_threat_hybrid"])

    def test_all_rules_accept_two_branches_and_reject_passive_lethality(self):
        self.assertTrue(all(permissions(context(bomb_branch_count=2), True, 300).values()))
        self.assertFalse(any(permissions(context(bomb_branch_count=3), False, 50).values()))


class StrictSelfTrapDefinitionTests(unittest.TestCase):
    def bomb(self, *, trace, death_cause=None, raw=True):
        return {
            "placed_step": 10,
            "viability_trace": trace,
            "death_cause": death_cause,
            "self_trapped_after_bomb": raw,
        }

    def test_placement_step_zero_is_not_a_strict_collapse(self):
        flags = strict_trap_flags(self.bomb(trace=[
            {"step": 10, "viable_action_count": 0},
            {"step": 11, "viable_action_count": 2},
        ]))
        self.assertTrue(flags["raw_zero"])
        self.assertFalse(flags["post_placement_escape_collapse"])
        self.assertFalse(flags["confirmed_catastrophic_self_trap"])

    def test_later_zero_without_own_bomb_death_is_not_confirmed(self):
        flags = strict_trap_flags(self.bomb(trace=[
            {"step": 11, "viable_action_count": 0},
        ], death_cause="opponent_bomb"))
        self.assertTrue(flags["post_placement_escape_collapse"])
        self.assertFalse(flags["confirmed_catastrophic_self_trap"])

    def test_later_zero_and_own_bomb_death_is_confirmed(self):
        flags = strict_trap_flags(self.bomb(trace=[
            {"step": 11, "viable_action_count": 1},
            {"step": 12, "viable_action_count": 0},
        ], death_cause="own_and_opponent_bomb"))
        self.assertTrue(flags["post_placement_escape_collapse"])
        self.assertTrue(flags["confirmed_catastrophic_self_trap"])
        self.assertEqual(flags["first_post_placement_zero_step"], 12)


if __name__ == "__main__":
    unittest.main()
