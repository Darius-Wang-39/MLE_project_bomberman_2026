import unittest
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from agent_code.dueling_ddqn_agent.callbacks import HUNT_CENTER_Q_BIAS, INPUT_DIM, migrate_policy_input
from agent_code.dueling_ddqn_agent.features import ACTIONS, FEATURE_NAMES, FeatureExtractor
from agent_code.dueling_ddqn_agent.model import DuelingQNetwork
from agent_code.dueling_ddqn_agent.replay import PrioritizedReplayBuffer, Transition
from agent_code.dueling_ddqn_agent.train import (
    EVENT_REWARDS,
    HUNT_REVISIT_PENALTY,
    LOW_VALUE_BOMB_PENALTY,
    SUICIDE_PENALTY_MULTIPLIER,
    _bomb_quality_penalty,
    _hunt_stall_penalty,
)
import events as e
from agent_code.dueling_ddqn_agent.symmetry import (
    D4_TRANSFORMS,
    transform_action_features,
    transform_action_index,
    transform_action_mask,
)
from tools.pretrain_behavior_clone import load_demonstrations, split_rounds


def sample_state():
    field = np.zeros((17, 17), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("me", 0, True, (2, 2)),
        "others": [],
        "bombs": [],
        "coins": [],
        "explosion_map": np.zeros_like(field, dtype=float),
    }


class DuelingDDQNTests(unittest.TestCase):
    def test_suicide_penalty_uses_requested_multiplier(self):
        self.assertEqual(SUICIDE_PENALTY_MULTIPLIER, 2.5)
        self.assertEqual(EVENT_REWARDS[e.KILLED_SELF], -50.0)

    def test_appended_center_feature_checkpoint_migration(self):
        target = DuelingQNetwork(INPUT_DIM, len(ACTIONS)).state_dict()
        new_feature_count = len(FEATURE_NAMES)
        old_feature_count = new_feature_count - 2
        old_width = len(ACTIONS) * old_feature_count
        old = {key: value.clone() for key, value in target.items()}
        old_weight = torch.arange(
            target["encoder.0.weight"].shape[0] * old_width,
            dtype=target["encoder.0.weight"].dtype,
        ).reshape(target["encoder.0.weight"].shape[0], old_width)
        old["encoder.0.weight"] = old_weight

        migrated, changed = migrate_policy_input(old, target)

        self.assertTrue(changed)
        for action_index in range(len(ACTIONS)):
            old_start = action_index * old_feature_count
            new_start = action_index * new_feature_count
            torch.testing.assert_close(
                migrated["encoder.0.weight"][:, new_start : new_start + old_feature_count],
                old_weight[:, old_start : old_start + old_feature_count],
            )
            self.assertEqual(
                torch.count_nonzero(
                    migrated["encoder.0.weight"][
                        :, new_start + old_feature_count : new_start + new_feature_count
                    ]
                ).item(),
                0,
            )

    def test_tactical_features_detect_trap_pressure_and_armed_threat(self):
        extractor = FeatureExtractor()
        state = sample_state()
        state["self"] = ("me", 0, True, (3, 2))
        state["others"] = [("enemy", 0, True, (2, 2))]
        state["field"][1, 2] = -1
        state["field"][2, 1] = -1
        state["field"][2, 3] = -1
        features = extractor.action_features(state)

        self.assertGreater(
            features[ACTIONS.index("BOMB"), FEATURE_NAMES.index("bomb_trap_pressure")],
            0.0,
        )
        self.assertGreater(
            features[ACTIONS.index("WAIT"), FEATURE_NAMES.index("armed_opponent_threat")],
            0.0,
        )

    def test_low_value_bomb_penalty_spares_productive_bombs(self):
        extractor = FeatureExtractor()
        trainer = SimpleNamespace(feature_extractor=extractor)
        empty = sample_state()
        self.assertEqual(
            _bomb_quality_penalty(trainer, empty, "BOMB"), LOW_VALUE_BOMB_PENALTY
        )

        productive = sample_state()
        productive["field"][3, 2] = 1
        self.assertEqual(_bomb_quality_penalty(trainer, productive, "BOMB"), 0.0)

    def test_hunt_center_progress_prefers_entering_central_box(self):
        extractor = FeatureExtractor()
        state = sample_state()
        state["self"] = ("me", 0, True, (2, 8))
        state["others"] = [("enemy", 0, True, (8, 8))]
        features = extractor.action_features(state)
        center_index = FEATURE_NAMES.index("hunt_center_progress")

        self.assertEqual(extractor.center_box_distance((2, 8)), 1)
        self.assertEqual(extractor.center_box_distance((3, 8)), 0)
        self.assertEqual(features[ACTIONS.index("RIGHT"), center_index], 1.0)
        self.assertEqual(features[ACTIONS.index("LEFT"), center_index], -1.0)
        self.assertGreater(HUNT_CENTER_Q_BIAS, 0.0)

    def test_endgame_potential_rewards_opponent_progress(self):
        extractor = FeatureExtractor()
        old_state = sample_state()
        old_state["field"] = np.zeros((17, 17), dtype=int)
        old_state["field"][[0, -1], :] = -1
        old_state["field"][:, [0, -1]] = -1
        old_state["coins"] = []
        old_state["self"] = ("me", 0, True, (2, 2))
        old_state["others"] = [("enemy", 0, True, (8, 2))]
        new_state = dict(old_state)
        new_state["self"] = ("me", 0, True, (3, 2))

        self.assertGreater(extractor.state_potential(new_state), extractor.state_potential(old_state))

    def test_hunt_stall_penalty_only_applies_to_recent_safe_revisit(self):
        extractor = FeatureExtractor()
        trainer = SimpleNamespace(
            position_history=deque(maxlen=8),
            history_round=None,
            feature_extractor=extractor,
        )
        old_state = sample_state()
        old_state["field"] = np.zeros((17, 17), dtype=int)
        old_state["field"][[0, -1], :] = -1
        old_state["field"][:, [0, -1]] = -1
        old_state["coins"] = []
        old_state["self"] = ("me", 0, True, (2, 2))
        old_state["others"] = [("enemy", 0, True, (8, 2))]
        new_state = dict(old_state)
        new_state["self"] = ("me", 0, True, (3, 2))

        self.assertEqual(_hunt_stall_penalty(trainer, old_state, new_state), 0.0)
        back_state = dict(old_state)
        back_state["self"] = ("me", 0, True, (2, 2))
        self.assertEqual(_hunt_stall_penalty(trainer, new_state, back_state), HUNT_REVISIT_PENALTY)

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
