import unittest
from types import SimpleNamespace
import numpy as np
import torch

from agent_code.ppo_agent.model import ActorCritic
from agent_code.ppo_agent.train import compute_gae, clipped_policy_loss


class PPOTests(unittest.TestCase):
    def test_mask_zeroes_illegal_probability(self):
        model = ActorCritic(4, 3)
        obs = torch.zeros((1, 4))
        mask = torch.tensor([[True, False, True]])
        dist, _ = model.distribution_and_value(obs, mask)
        self.assertEqual(float(dist.probs[0, 1].detach()), 0.0)
        self.assertAlmostEqual(float(dist.probs.sum().detach()), 1.0, places=6)

    def test_fresh_policy_ratio_is_one(self):
        model = ActorCritic(4, 3)
        obs = torch.zeros((2, 4))
        mask = torch.ones((2, 3), dtype=torch.bool)
        dist, _ = model.distribution_and_value(obs, mask)
        actions = torch.tensor([0, 2])
        old = dist.log_prob(actions).detach()
        ratio = torch.exp(dist.log_prob(actions) - old)
        torch.testing.assert_close(ratio, torch.ones_like(ratio))

    def test_terminal_gae_does_not_bootstrap(self):
        advantages, returns = compute_gae([2.0], [0.5], [99.0], [True])
        self.assertAlmostEqual(float(advantages[0]), 1.5)
        self.assertAlmostEqual(float(returns[0]), 2.0)

    def test_two_step_gae_matches_hand_calculation(self):
        advantages, _ = compute_gae([1.0, 2.0], [0.5, 0.25], [0.25, 0.0], [False, True])
        last = 2.0 - 0.25
        first = 1.0 + 0.99 * 0.25 - 0.5 + 0.99 * 0.95 * last
        self.assertAlmostEqual(float(advantages[1]), last, places=5)
        self.assertAlmostEqual(float(advantages[0]), first, places=5)

    def test_clipping_for_positive_advantage(self):
        ratio = torch.tensor([2.0])
        advantage = torch.tensor([1.0])
        self.assertAlmostEqual(float(clipped_policy_loss(ratio, advantage)), -1.2, places=6)

    def test_clipping_for_negative_advantage(self):
        ratio = torch.tensor([0.2])
        advantage = torch.tensor([-1.0])
        self.assertAlmostEqual(float(clipped_policy_loss(ratio, advantage)), 0.8, places=6)


if __name__ == "__main__":
    unittest.main()
