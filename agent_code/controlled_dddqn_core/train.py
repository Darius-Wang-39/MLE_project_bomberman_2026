"""DDDQN training with an isolated D4 switch and shared reward."""

from __future__ import annotations

import numpy as np

from agent_code.controlled_common.reward import transition_reward
from agent_code.controlled_common.seeding import numpy_rng
from agent_code.dueling_ddqn_agent import train as _base
from agent_code.dueling_ddqn_agent.symmetry import (
    D4_TRANSFORMS,
    transform_action_features,
    transform_action_index,
    transform_action_mask,
)
from .callbacks import ACTIONS, MODEL_FILE, vectorize_state

_augmentation_rng = None
_d4_enabled = False


def _controlled_augment(transitions, ignored_rng):
    del ignored_rng
    feature_count = transitions[0].state.size // len(ACTIONS)
    output = ([], [], [], [], [], [], [])
    for transition in transitions:
        transform = (
            D4_TRANSFORMS[int(_augmentation_rng.integers(len(D4_TRANSFORMS)))]
            if _d4_enabled
            else D4_TRANSFORMS[0]
        )
        output[0].append(transform_action_features(transition.state, transform, feature_count))
        output[1].append(transform_action_index(transition.action, transform))
        output[2].append(transition.reward)
        output[3].append(transform_action_features(transition.next_state, transform, feature_count))
        output[4].append(transform_action_mask(transition.next_mask, transform))
        output[5].append(transition.done)
        output[6].append(transition.discount)
    return output


def setup_training(self):
    global _augmentation_rng, _d4_enabled
    _base.setup_training(self)
    self.replay_rng = numpy_rng(51)
    _augmentation_rng = numpy_rng(52)
    _d4_enabled = bool(self.d4_enabled)


_base.ACTIONS = ACTIONS
_base.MODEL_FILE = MODEL_FILE
_base.vectorize_state = vectorize_state
_base._reward = transition_reward
_base._augment_batch = _controlled_augment

game_events_occurred = _base.game_events_occurred
end_of_round = _base.end_of_round
