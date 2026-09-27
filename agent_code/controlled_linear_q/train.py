"""Reuse the original n-step learner with the controlled callback/model path."""

from agent_code.linear_q_agent import train as _base
from .callbacks import ACTIONS, MODEL_FILE

_base.ACTIONS = ACTIONS
_base.MODEL_FILE = MODEL_FILE

setup_training = _base.setup_training
game_events_occurred = _base.game_events_occurred
end_of_round = _base.end_of_round
