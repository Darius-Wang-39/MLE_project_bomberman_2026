import unittest

import numpy as np
import torch

from agent_code.dueling_ddqn_agent.callbacks import INPUT_DIM
from agent_code.dueling_ddqn_agent.features import ACTIONS, FEATURE_NAMES
from agent_code.dueling_ddqn_agent.model import DuelingQNetwork
from agent_code.dueling_ddqn_agent.replay import PrioritizedReplayBuffer, Transition
from agent_code.dueling_ddqn_agent.symmetry import (
    D4_TRANSFORMS,
    transform_action_features,
    transform_action_index,
    transform_action_mask,
)


class DuelingDDQNTests(unittest.TestCase):
    def test_network_output_shape(self):
        network = DuelingQNetwork(INPUT_DIM, len(ACTIONS))
        output = network(torch.zeros(3, INPUT_DIM))
        self.assertEqual(tuple(output.shape), (3, len(ACTIONS)))

    def test_prioritized_replay_round_trip(self):
        buffer = PrioritizedReplayBuffer(16)
        transition = Transition(
            state=np.zeros(INPUT_DIM, dtype=np.float32),
            action=1,
            reward=2.0,
            next_state=np.ones(INPUT_DIM, dtype=np.float32),
            next_mask=np.ones(len(ACTIONS), dtype=bool),
            done=False,
            discount=0.8,
        )
        for _ in range(8):
            buffer.add(transition)
        sampled, indices, weights = buffer.sample(4, beta=0.4, rng=np.random.default_rng(7))
        self.assertEqual(len(sampled), 4)
        self.assertEqual(weights.shape, (4,))
        buffer.update_priorities(indices, np.full(4, 3.0))
        self.assertTrue(np.all(buffer.priorities[indices] == 3.0))

    def test_action_feature_symmetry_permutation(self):
        rows = np.arange(len(ACTIONS) * len(FEATURE_NAMES)).reshape(len(ACTIONS), len(FEATURE_NAMES))
        mask = np.array([True, False, True, False, True, True])
        for transform in D4_TRANSFORMS:
            transformed = transform_action_features(rows.reshape(-1), transform, len(FEATURE_NAMES)).reshape(rows.shape)
            transformed_mask = transform_action_mask(mask, transform)
            for index in range(len(ACTIONS)):
                mapped = transform_action_index(index, transform)
                np.testing.assert_array_equal(transformed[mapped], rows[index])
                self.assertEqual(bool(transformed_mask[mapped]), bool(mask[index]))


if __name__ == "__main__":
    unittest.main()

