"""Use the current DDQN representation and safety mask unchanged."""

from agent_code.dueling_ddqn_agent.features import (
    ACTIONS, FEATURE_NAMES, MOVE_DELTAS, FeatureExtractor,
)

__all__ = ["ACTIONS", "FEATURE_NAMES", "MOVE_DELTAS", "FeatureExtractor"]
