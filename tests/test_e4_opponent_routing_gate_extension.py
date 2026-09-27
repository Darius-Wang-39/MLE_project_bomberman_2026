import unittest

from evaluation import e4_opponent_routing_gate as base
from evaluation import e4_opponent_routing_gate_extension as extension


class OpponentRoutingExtensionTests(unittest.TestCase):
    def test_extension_only_changes_lineups_and_result_root(self):
        original_variants = base.VARIANTS
        original_seeds = base.SEEDS
        original_rounds = base.ROUNDS_PER_SEED
        spec = extension.extension_protocol()
        self.assertEqual(base.VARIANTS, original_variants)
        self.assertEqual(base.SEEDS, original_seeds)
        self.assertEqual(base.ROUNDS_PER_SEED, original_rounds)
        self.assertEqual(spec["lineups"], extension.LINEUPS)
        self.assertEqual(spec["total_rounds"], 1200)
        self.assertEqual(extension.LINEUPS["rule3"], ["rule_based_agent"] * 3)


if __name__ == "__main__":
    unittest.main()
