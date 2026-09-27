import json
from pathlib import Path

import numpy as np

from agent_code.controlled_common.features import ACTIONS, FEATURE_NAMES
from agent_code.controlled_dddqn_core import train as controlled_train
from agent_code.dueling_ddqn_agent.replay import Transition


ROOT = Path(__file__).resolve().parents[1]


class FixedTransformRng:
    def __init__(self, index):
        self.index = index

    def integers(self, upper):
        assert upper == 8
        return self.index


def transition():
    size = len(ACTIONS) * len(FEATURE_NAMES)
    return Transition(
        state=np.arange(size, dtype=np.float32),
        action=0,
        reward=1.25,
        next_state=np.arange(size, dtype=np.float32) + 1000,
        next_mask=np.asarray([True, False, True, False, True, True]),
        done=False,
        discount=0.95,
    )


def test_controlled_protocol_has_five_models_and_three_tiers():
    config = json.loads(
        (ROOT / "experiments" / "configs" / "controlled_family_24h.json").read_text()
    )
    assert len(config["models"]) == 5
    assert config["models"][-2:] == ["controlled_dddqn_nod4", "controlled_dddqn_d4"]
    assert config["tiers"] == {"1": 1, "2": 2, "3": 3}
    assert sum(stage["rounds"] for stage in config["phase_a"]) == 380
    assert config["task4_branches"] == [25, 50, 100]


def test_controlled_dddqn_identity_path_does_not_augment():
    sample = transition()
    controlled_train._d4_enabled = False
    controlled_train._augmentation_rng = FixedTransformRng(7)
    states, actions, rewards, next_states, masks, dones, discounts = (
        controlled_train._controlled_augment([sample], ignored_rng=None)
    )
    np.testing.assert_array_equal(states[0], sample.state)
    np.testing.assert_array_equal(next_states[0], sample.next_state)
    np.testing.assert_array_equal(masks[0], sample.next_mask)
    assert actions == [sample.action]
    assert rewards == [sample.reward]
    assert dones == [sample.done]
    assert discounts == [sample.discount]


def test_controlled_dddqn_d4_path_transforms_same_sample():
    sample = transition()
    controlled_train._d4_enabled = True
    controlled_train._augmentation_rng = FixedTransformRng(1)
    states, actions, _, next_states, masks, _, _ = controlled_train._controlled_augment(
        [sample], ignored_rng=None
    )
    assert not np.array_equal(states[0], sample.state)
    assert not np.array_equal(next_states[0], sample.next_state)
    assert actions[0] != sample.action
    assert not np.array_equal(masks[0], sample.next_mask)
