"""Rule-based demonstration collector used only for offline pretraining."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

from agent_code.dueling_ddqn_agent.callbacks import vectorize_state
from agent_code.dueling_ddqn_agent.features import ACTIONS, FeatureExtractor
from agent_code.rule_based_agent import callbacks as rule_agent


def setup(self):
    seed = int(os.environ.get("BOMBERMAN_DEMO_SEED", "5001"))
    rule_agent.setup(self)
    random.seed(seed)
    np.random.seed(seed)
    self.demo_extractor = FeatureExtractor()
    self.demo_path = Path(os.environ.get("BOMBERMAN_DEMO_PATH", "demonstrations.npz"))
    self.demo_states = []
    self.demo_actions = []
    self.demo_safe_masks = []
    self.demo_rounds = []
    self.demo_steps = []
    self.demo_fallback_count = 0


def act(self, game_state: dict) -> str:
    action = rule_agent.act(self, game_state)
    safe_mask = self.demo_extractor.safe_action_mask(game_state)
    if action not in ACTIONS:
        self.demo_fallback_count += 1
        safe_indices = np.flatnonzero(safe_mask)
        wait_index = ACTIONS.index("WAIT")
        action = ACTIONS[
            wait_index if safe_mask[wait_index] else int(safe_indices[0])
        ] if safe_indices.size else "WAIT"
    self.demo_states.append(vectorize_state(self.demo_extractor, game_state))
    self.demo_actions.append(ACTIONS.index(action))
    self.demo_safe_masks.append(safe_mask)
    self.demo_rounds.append(int(game_state["round"]))
    self.demo_steps.append(int(game_state["step"]))
    return action


def save_demonstrations(self):
    """Atomically persist all demonstrations collected by this agent process."""
    if not self.demo_states:
        return
    self.demo_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = self.demo_path.with_suffix(self.demo_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            states=np.asarray(self.demo_states, dtype=np.float32),
            actions=np.asarray(self.demo_actions, dtype=np.int64),
            safe_masks=np.asarray(self.demo_safe_masks, dtype=bool),
            rounds=np.asarray(self.demo_rounds, dtype=np.int32),
            steps=np.asarray(self.demo_steps, dtype=np.int32),
            fallback_count=np.asarray(self.demo_fallback_count, dtype=np.int32),
        )
    temporary.replace(self.demo_path)
