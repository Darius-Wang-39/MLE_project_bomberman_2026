import importlib
import unittest
from types import SimpleNamespace

import numpy as np


model_module = importlib.import_module("agent_code.sarsa-lambda.model")
train = importlib.import_module("agent_code.sarsa-lambda.train")


class SarsaLambdaTests(unittest.TestCase):
    def test_update_matches_accumulating_trace_equations(self):
        agent = SimpleNamespace(
            model=model_module.SarsaLambdaModel(3),
            eligibility=np.zeros(3),
            last_td_error=0.0,
            episode_td_errors=[],
        )
        phi = np.array([1.0, 0.5, 0.0])
        phi_next = np.array([1.0, 0.0, 0.5])
        train._update(agent, phi, 2.0, phi_next)
        self.assertAlmostEqual(agent.last_td_error, 2.0)
        np.testing.assert_allclose(agent.eligibility, phi)
        np.testing.assert_allclose(agent.model.weights, train.ALPHA * 2.0 * phi)

    def test_terminal_update_has_no_bootstrap(self):
        agent = SimpleNamespace(
            model=model_module.SarsaLambdaModel(2, weights=np.array([1.0, 0.0])),
            eligibility=np.zeros(2),
            last_td_error=0.0,
            episode_td_errors=[],
        )
        train._update(agent, np.array([1.0, 0.0]), -2.0, None)
        self.assertAlmostEqual(agent.last_td_error, -3.0)


if __name__ == "__main__":
    unittest.main()
