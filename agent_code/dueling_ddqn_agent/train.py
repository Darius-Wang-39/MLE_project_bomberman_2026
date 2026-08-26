"""Dueling Double DQN with n-step returns and prioritized replay."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn

import events as e
from .callbacks import ACTIONS, MODEL_FILE, vectorize_state
from .model import DuelingQNetwork
from .replay import PrioritizedReplayBuffer, Transition
from .symmetry import (
    D4_TRANSFORMS,
    transform_action_features,
    transform_action_index,
    transform_action_mask,
)


GAMMA = 0.95
N_STEP = 5
BATCH_SIZE = 64
REPLAY_CAPACITY = 100_000
REPLAY_WARMUP = 1_000
TRAIN_EVERY = 4
LEARNING_RATE = 3e-4
TAU = 0.005
SHAPING_SCALE = 0.5
HUNT_REVISIT_PENALTY = -0.12
HUNT_HISTORY_LENGTH = 8
MIDGAME_PRODUCTIVE_REVISIT_PENALTY = -0.20
LOW_VALUE_BOMB_PENALTY = -0.25
SAFE_PRODUCTIVE_BOMB_REWARD = 0.30
TRAP_PRESSURE_THRESHOLD = 0.15
SUICIDE_PENALTY_MULTIPLIER = 2.5
PRIORITY_BETA_START = 0.4
PRIORITY_BETA_STEPS = 250_000

EVENT_REWARDS = {
    e.MOVED_LEFT: -0.02,
    e.MOVED_RIGHT: -0.02,
    e.MOVED_UP: -0.02,
    e.MOVED_DOWN: -0.02,
    e.WAITED: -0.15,
    e.INVALID_ACTION: -2.0,
    e.BOMB_DROPPED: -0.02,
    e.CRATE_DESTROYED: 0.8,
    e.COIN_FOUND: 0.5,
    e.COIN_COLLECTED: 4.0,
    e.KILLED_OPPONENT: 10.0,
    e.KILLED_SELF: -20.0 * SUICIDE_PENALTY_MULTIPLIER,
    e.GOT_KILLED: -15.0,
    e.SURVIVED_ROUND: 2.0,
}


def setup_training(self):
    self.policy_net.train()
    self.target_net = DuelingQNetwork(self.policy_net.encoder[0].in_features, len(ACTIONS)).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()
    self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=LEARNING_RATE)
    if self.loaded_optimizer_state is not None:
        self.optimizer.load_state_dict(self.loaded_optimizer_state)
    self.loss_fn = nn.SmoothL1Loss(reduction="none")
    self.replay = PrioritizedReplayBuffer(REPLAY_CAPACITY)
    self.n_step_buffer = deque()
    self.replay_rng = np.random.default_rng()
    self.position_history = deque(maxlen=HUNT_HISTORY_LENGTH)
    self.history_round = None


def _hunt_stall_penalty(self, old_game_state, action: str, new_game_state) -> float:
    """Penalize safe loops in endgame or beside a productive bomb target."""
    if new_game_state is None:
        return 0.0

    round_id = old_game_state["round"]
    old_position = tuple(old_game_state["self"][-1])
    new_position = tuple(new_game_state["self"][-1])
    if self.history_round != round_id:
        self.position_history.clear()
        self.position_history.append(old_position)
        self.history_round = round_id

    hunt_phase = (
        not old_game_state["coins"]
        and not np.any(old_game_state["field"] == 1)
        and bool(old_game_state["others"])
    )
    in_danger = (
        bool(old_game_state["bombs"])
        or old_game_state["explosion_map"][old_position] > 0
    )
    visits = list(self.position_history).count(new_position)
    self.position_history.append(new_position)
    if in_danger or visits == 0 or action == "BOMB":
        return 0.0

    if hunt_phase:
        return HUNT_REVISIT_PENALTY * min(visits, 3)

    bomb_index = ACTIONS.index("BOMB")
    bomb_safe = bool(self.feature_extractor.safe_action_mask(old_game_state)[bomb_index])
    field = old_game_state["field"]
    blast = self.feature_extractor.blast_tiles(field, old_position)
    productive = any(field[tile] == 1 for tile in blast)
    if bomb_safe and productive:
        return MIDGAME_PRODUCTIVE_REVISIT_PENALTY * min(visits, 3)
    return 0.0


def _bomb_quality_penalty(self, old_game_state, action: str) -> float:
    """Discourage bombs that neither farm resources nor restrict an opponent."""
    if action != "BOMB":
        return 0.0
    bomb_row = self.feature_extractor.action_features(old_game_state)[ACTIONS.index("BOMB")]
    crate_yield = bomb_row[self.feature_extractor.feature_names.index("bomb_crate_yield")]
    opponent_yield = bomb_row[self.feature_extractor.feature_names.index("bomb_opponent_yield")]
    trap_pressure = bomb_row[self.feature_extractor.feature_names.index("bomb_trap_pressure")]
    safe_productive = bomb_row[
        self.feature_extractor.feature_names.index("safe_productive_bomb")
    ]
    if safe_productive > 0.0:
        return SAFE_PRODUCTIVE_BOMB_REWARD
    if crate_yield > 0.0 or opponent_yield > 0.0 or trap_pressure >= TRAP_PRESSURE_THRESHOLD:
        return 0.0
    return LOW_VALUE_BOMB_PENALTY


def _reward(self, old_game_state, action: str, new_game_state, events: List[str]) -> float:
    event_reward = float(sum(EVENT_REWARDS.get(event, 0.0) for event in events))
    shaping = SHAPING_SCALE * (
        GAMMA * self.feature_extractor.state_potential(new_game_state)
        - self.feature_extractor.state_potential(old_game_state)
    )
    return (
        event_reward
        + shaping
        + _hunt_stall_penalty(self, old_game_state, action, new_game_state)
        + _bomb_quality_penalty(self, old_game_state, action)
    )


def _emit_n_step(self, count: int):
    count = min(count, len(self.n_step_buffer))
    first = self.n_step_buffer[0]
    last = self.n_step_buffer[count - 1]
    reward = float(sum((GAMMA ** index) * self.n_step_buffer[index].reward for index in range(count)))
    self.replay.add(
        Transition(
            state=first.state,
            action=first.action,
            reward=reward,
            next_state=last.next_state,
            next_mask=last.next_mask,
            done=last.done,
            discount=GAMMA ** count,
        )
    )
    self.n_step_buffer.popleft()


def _record(self, old_game_state, action: str, new_game_state, events: List[str]):
    state = vectorize_state(self.feature_extractor, old_game_state)
    next_state = (
        np.zeros_like(state)
        if new_game_state is None
        else vectorize_state(self.feature_extractor, new_game_state)
    )
    next_mask = (
        np.zeros(len(ACTIONS), dtype=bool)
        if new_game_state is None
        else self.feature_extractor.safe_action_mask(new_game_state)
    )
    raw = Transition(
        state=state,
        action=ACTIONS.index(action),
        reward=_reward(self, old_game_state, action, new_game_state, events),
        next_state=next_state,
        next_mask=next_mask,
        done=new_game_state is None,
        discount=GAMMA,
    )
    self.n_step_buffer.append(raw)
    if raw.done:
        while self.n_step_buffer:
            _emit_n_step(self, min(N_STEP, len(self.n_step_buffer)))
    elif len(self.n_step_buffer) >= N_STEP:
        _emit_n_step(self, N_STEP)


def _augment_batch(transitions, rng):
    feature_count = transitions[0].state.size // len(ACTIONS)
    states, actions, rewards, next_states, next_masks, dones, discounts = [], [], [], [], [], [], []
    for transition in transitions:
        transform = D4_TRANSFORMS[int(rng.integers(len(D4_TRANSFORMS)))]
        states.append(transform_action_features(transition.state, transform, feature_count))
        actions.append(transform_action_index(transition.action, transform))
        rewards.append(transition.reward)
        next_states.append(transform_action_features(transition.next_state, transform, feature_count))
        next_masks.append(transform_action_mask(transition.next_mask, transform))
        dones.append(transition.done)
        discounts.append(transition.discount)
    return states, actions, rewards, next_states, next_masks, dones, discounts


def _priority_beta(self) -> float:
    fraction = min(self.learner_updates / PRIORITY_BETA_STEPS, 1.0)
    return PRIORITY_BETA_START + fraction * (1.0 - PRIORITY_BETA_START)


def optimize_model(self):
    if len(self.replay) < max(BATCH_SIZE, REPLAY_WARMUP):
        return
    if self.environment_steps % TRAIN_EVERY != 0:
        return

    transitions, indices, importance = self.replay.sample(BATCH_SIZE, _priority_beta(self), self.replay_rng)
    states, actions, rewards, next_states, next_masks, dones, discounts = _augment_batch(transitions, self.replay_rng)

    states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
    actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
    next_states_t = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
    next_masks_t = torch.as_tensor(np.asarray(next_masks), dtype=torch.bool, device=self.device)
    dones_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
    discounts_t = torch.as_tensor(discounts, dtype=torch.float32, device=self.device)
    importance_t = torch.as_tensor(importance, dtype=torch.float32, device=self.device)

    predicted = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
    with torch.no_grad():
        policy_next = self.policy_net(next_states_t).masked_fill(~next_masks_t, -torch.inf)
        next_actions = policy_next.argmax(dim=1, keepdim=True)
        target_next = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
        target = rewards_t + (~dones_t).float() * discounts_t * target_next

    td_error = target - predicted
    loss = (importance_t * self.loss_fn(predicted, target)).mean()
    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
    self.optimizer.step()
    self.replay.update_priorities(indices, td_error.detach().abs().cpu().numpy() + 1e-5)

    with torch.no_grad():
        for target_parameter, policy_parameter in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_parameter.mul_(1.0 - TAU).add_(policy_parameter, alpha=TAU)
    self.learner_updates += 1


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    _record(self, old_game_state, self_action, new_game_state, events)
    self.environment_steps += 1
    optimize_model(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    _record(self, last_game_state, last_action, None, events)
    self.environment_steps += 1
    optimize_model(self)
    checkpoint = {
        "policy_state": self.policy_net.state_dict(),
        "optimizer_state": self.optimizer.state_dict(),
        "environment_steps": self.environment_steps,
        "learner_updates": self.learner_updates,
    }
    temporary = Path(MODEL_FILE + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(MODEL_FILE)
