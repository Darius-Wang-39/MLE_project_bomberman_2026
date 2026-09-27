"""Controlled PPO using the common reward and 17-feature observation."""

from agent_code.controlled_common.reward import transition_reward
from agent_code.ppo_agent import train as _base
from .callbacks import ACTIONS, MODEL_FILE, vectorize_state

_base.ACTIONS = ACTIONS
_base.MODEL_FILE = MODEL_FILE
_base.vectorize_state = vectorize_state
_base.ddqn_reward = transition_reward

setup_training = _base.setup_training
game_events_occurred = _base.game_events_occurred
end_of_round = _base.end_of_round
