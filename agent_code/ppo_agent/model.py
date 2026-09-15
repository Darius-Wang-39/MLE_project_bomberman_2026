"""Small shared-backbone actor-critic network."""

import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_count):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.actor = nn.Linear(128, action_count)
        self.critic = nn.Linear(128, 1)

    def forward(self, observations):
        hidden = self.backbone(observations)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def distribution_and_value(self, observations, masks):
        logits, values = self(observations)
        masked_logits = logits.masked_fill(~masks, -1e9)
        return Categorical(logits=masked_logits), values
