"""Lambda=0.7 training branch."""
import importlib
_base = importlib.import_module("agent_code.sarsa-lambda.train")
setup_training = _base.setup_training
game_events_occurred = _base.game_events_occurred
end_of_round = _base.end_of_round
