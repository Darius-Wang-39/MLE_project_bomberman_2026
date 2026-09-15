import json
import tempfile
import unittest
from pathlib import Path

from evaluation.summarize_stats import summarize, summarize_match
from evaluation.run_locked_suite import build_command


class SummaryTests(unittest.TestCase):
    def test_locked_suite_requires_three_opponents(self):
        lineup = {"opponents": ["random_agent"], "rounds": 2, "seed": 3}
        with self.assertRaises(ValueError):
            build_command("candidate", lineup, Path("out.json"))

    def test_weighted_summary(self):
        payload = {
            "by_agent": {
                "linear_q_agent": {
                    "rounds": 2,
                    "steps": 100,
                    "score": 6,
                    "coins": 6,
                    "time": 0.2,
                    "suicides": 1,
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = summarize([path], "linear_q_agent")
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["mean_score"], 3.0)
        self.assertEqual(result["suicides_per_round"], 0.5)
        self.assertEqual(result["mean_action_ms"], 2.0)

    def test_match_summary_handles_ties_rank_and_margin(self):
        payload = {
            "by_round": {
                "round_1": {"agents": {
                    "candidate": {"score": 5, "coins": 5, "survived": True},
                    "opponent_0": {"score": 5, "survived": True},
                    "opponent_1": {"score": 1, "survived": False},
                    "opponent_2": {"score": 0, "survived": False},
                }},
                "round_2": {"agents": {
                    "candidate": {"score": 2, "coins": 2, "survived": True},
                    "opponent_0": {"score": 3, "survived": True},
                    "opponent_1": {"score": 1, "survived": False},
                    "opponent_2": {"score": 0, "survived": False},
                }},
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = {row["agent"]: row for row in summarize_match([path])}["candidate"]
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["mean_score"], 3.5)
        self.assertEqual(result["win_rate"], 0.5)
        self.assertEqual(result["mean_rank"], 1.5)
        self.assertAlmostEqual(result["mean_score_margin"], 11 / 6)
        self.assertEqual(result["survival_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
