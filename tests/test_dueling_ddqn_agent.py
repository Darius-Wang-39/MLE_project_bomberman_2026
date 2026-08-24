import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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
from tools.pretrain_behavior_clone import load_demonstrations, split_rounds


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

    def test_demonstration_round_split_has_no_leakage(self):
        rounds = np.repeat(np.arange(10), 3)
        train_mask, validation_mask = split_rounds(rounds, 0.2, seed=7)
        self.assertFalse(set(rounds[train_mask]) & set(rounds[validation_mask]))
        self.assertEqual(int(np.sum(train_mask) + np.sum(validation_mask)), len(rounds))

    def test_demonstration_loader_validates_shape(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "demo.npz"
            np.savez(
                path,
                states=np.zeros((4, INPUT_DIM), dtype=np.float32),
                actions=np.zeros(4, dtype=np.int64),
                safe_masks=np.ones((4, len(ACTIONS)), dtype=bool),
                rounds=np.array([1, 1, 2, 2], dtype=np.int32),
            )
            states, actions, masks, rounds = load_demonstrations(path)
            self.assertEqual(states.shape, (4, INPUT_DIM))
            self.assertEqual(actions.shape, (4,))
            self.assertEqual(masks.shape, (4, len(ACTIONS)))
            self.assertEqual(rounds.tolist(), [1, 1, 2, 2])


if __name__ == "__main__":
    unittest.main()
