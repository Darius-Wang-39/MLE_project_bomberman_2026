"""Day 9 ablation: use immediate legality without survival lookahead."""

from __future__ import annotations

import numpy as np
import torch

from agent_code.dueling_ddqn_agent.callbacks import setup, vectorize_state
from agent_code.dueling_ddqn_agent.features import ACTIONS


def act(self, game_state: dict) -> str:
    allowed = self.feature_extractor.legal_action_mask(game_state)
    allowed_indices = np.flatnonzero(allowed)
    if allowed_indices.size == 0:
        return "WAIT"
    state = torch.from_numpy(
        vectorize_state(self.feature_extractor, game_state)
    ).unsqueeze(0).to(self.device)
    with torch.no_grad():
        q_values = self.policy_net(state).squeeze(0).cpu().numpy()
    masked_q = np.where(allowed, q_values, -np.inf)
    best = np.flatnonzero(np.isclose(masked_q, np.max(masked_q)))
    return ACTIONS[self.rng.choice(best.tolist())]

