"""Current validation-selected SARSA(lambda) inference agent."""
import importlib
_base = importlib.import_module("agent_code.sarsa-lambda.callbacks")
setup, act, state_to_features = _base.setup, _base.act, _base.state_to_features
