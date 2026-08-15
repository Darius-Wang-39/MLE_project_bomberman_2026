import json
import tempfile
import unittest
from pathlib import Path

from evaluation.summarize_stats import summarize


class SummaryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

