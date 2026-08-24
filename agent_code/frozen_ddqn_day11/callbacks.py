"""Frozen Day 11 baseline for the central-endgame ablation."""

from agent_code.dueling_ddqn_agent.callbacks import act, setup as shared_setup


def setup(self):
    shared_setup(self)
    self.hunt_center_q_bias = 0.0
