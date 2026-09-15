"""Shared action-relative features for the SARSA(lambda) agent."""

# Keeping this as a single import makes the SARSA/DDQN comparison use the same
# feature semantics and safety mask as the established linear baseline.
from agent_code.linear_q_agent.features import ACTIONS, FEATURE_NAMES, FeatureExtractor

__all__ = ["ACTIONS", "FEATURE_NAMES", "FeatureExtractor"]
