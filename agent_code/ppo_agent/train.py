"""On-policy rollout collection, GAE, and clipped PPO updates."""

from collections import deque
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn.functional as F

from .callbacks import ACTIONS, MODEL_FILE, vectorize_state
from .config import (
    GAMMA, GAE_LAMBDA, LEARNING_RATE, ROLLOUT_STEPS, UPDATE_EPOCHS,
    MINIBATCH_SIZE, CLIP_EPSILON, VALUE_COEF, ENTROPY_COEF,
    MAX_GRAD_NORM, TARGET_KL,
)
from agent_code.dueling_ddqn_agent.train import _reward as ddqn_reward


def setup_training(self):
    self.model.train()
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
    if self.loaded_optimizer_state is not None:
        self.optimizer.load_state_dict(self.loaded_optimizer_state)
    self.rollout = []
    self.position_history = deque(maxlen=32)
    self.history_round = None
    self.training_metrics = []


def compute_gae(rewards, values, next_values, dones, gamma=GAMMA, gae_lambda=GAE_LAMBDA):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        nonterminal = 1.0 - float(dones[index])
        delta = rewards[index] + gamma * next_values[index] * nonterminal - values[index]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[index] = gae
    return advantages, advantages + np.asarray(values, dtype=np.float32)


def clipped_policy_loss(ratios, advantages, clip_epsilon=CLIP_EPSILON):
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    return -torch.minimum(unclipped, clipped).mean()


def _next_value(self, game_state):
    if game_state is None:
        return 0.0
    obs = torch.from_numpy(vectorize_state(self.feature_extractor, game_state)).unsqueeze(0).to(self.device)
    with torch.no_grad():
        _, value = self.model(obs)
    return float(value.item())


def _update(self):
    if not self.rollout:
        return
    obs, actions, rewards, dones, old_log_probs, values, masks, next_values = zip(*self.rollout)
    advantages, returns = compute_gae(rewards, values, next_values, dones)
    return_variance = float(np.var(returns))
    explained_variance = (
        0.0 if return_variance < 1e-8
        else 1.0 - float(np.var(returns - np.asarray(values, dtype=np.float32))) / return_variance
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    tensors = (
        torch.from_numpy(np.asarray(obs)).float().to(self.device),
        torch.tensor(actions, dtype=torch.long, device=self.device),
        torch.tensor(old_log_probs, dtype=torch.float32, device=self.device),
        torch.from_numpy(advantages).to(self.device),
        torch.from_numpy(returns).to(self.device),
        torch.from_numpy(np.asarray(masks)).bool().to(self.device),
    )
    count = len(actions)
    latest = {}
    stop_early = False
    for _ in range(UPDATE_EPOCHS):
        for indices in torch.randperm(count, device=self.device).split(MINIBATCH_SIZE):
            batch_obs, batch_actions, batch_old_logp, batch_adv, batch_returns, batch_masks = (
                tensor[indices] for tensor in tensors
            )
            dist, predicted_values = self.model.distribution_and_value(batch_obs, batch_masks)
            new_logp = dist.log_prob(batch_actions)
            ratios = torch.exp(new_logp - batch_old_logp)
            policy_loss = clipped_policy_loss(ratios, batch_adv)
            value_loss = F.mse_loss(predicted_values, batch_returns)
            entropy = dist.entropy().mean()
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()
            with torch.no_grad():
                approx_kl = float((batch_old_logp - new_logp).mean().item())
                clip_fraction = float(((ratios - 1.0).abs() > CLIP_EPSILON).float().mean().item())
            latest = dict(policy_loss=float(policy_loss.item()), value_loss=float(value_loss.item()),
                          entropy=float(entropy.item()), approx_kl=approx_kl,
                          clip_fraction=clip_fraction, explained_variance=explained_variance)
            self.learner_updates += 1
            if approx_kl > TARGET_KL:
                stop_early = True
                break
        if stop_early:
            break
    self.training_metrics.append(latest)
    self.rollout.clear()


def _record(self, old_state, action, new_state, events, done):
    if self.pending_action is None:
        raise RuntimeError("PPO act data missing for transition")
    obs, action_index, log_prob, value, mask = self.pending_action
    if ACTIONS[action_index] != action:
        raise RuntimeError("PPO action callback mismatch")
    reward = ddqn_reward(self, old_state, action, new_state, events)
    self.rollout.append((obs, action_index, reward, done, log_prob, value, mask, _next_value(self, new_state)))
    self.pending_action = None
    self.environment_steps += 1
    if len(self.rollout) >= ROLLOUT_STEPS or done:
        _update(self)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events: List[str]):
    _record(self, old_game_state, self_action, new_game_state, events, False)


def end_of_round(self, last_game_state, last_action, events: List[str]):
    if self.pending_action is not None:
        _record(self, last_game_state, last_action, None, events, True)
    elif self.rollout:
        # The framework may report the final step through game_events_occurred
        # before end_of_round. Convert that already-recorded transition to a
        # terminal one so GAE cannot bootstrap across the episode boundary.
        transition = list(self.rollout[-1])
        transition[3] = True
        transition[7] = 0.0
        self.rollout[-1] = tuple(transition)
        _update(self)
    torch.save(dict(model_state=self.model.state_dict(), optimizer_state=self.optimizer.state_dict(),
                    environment_steps=self.environment_steps, learner_updates=self.learner_updates,
                    latest_metrics=self.training_metrics[-1] if self.training_metrics else {}), MODEL_FILE)
