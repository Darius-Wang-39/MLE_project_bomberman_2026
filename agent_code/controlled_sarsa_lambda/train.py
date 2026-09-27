"""Reuse SARSA(lambda), replacing its extra bomb reward with the common reward."""

import importlib

from agent_code.controlled_common.reward import transition_reward
from .callbacks import ACTIONS, MODEL_FILE, exploration_rate

_base = importlib.import_module("agent_code.sarsa-lambda.train")

_base.ACTIONS = ACTIONS
_base.MODEL_FILE = MODEL_FILE
_base.exploration_rate = exploration_rate
_base._transition_reward = transition_reward

setup_training = _base.setup_training
complete_pending_update = _base.complete_pending_update
game_events_occurred = _base.game_events_occurred
end_of_round = _base.end_of_round
