"""Masked sampling during PPO training and deterministic masked inference."""

from pathlib import Path
import numpy as np
import torch

from .features import ACTIONS, FEATURE_NAMES, FeatureExtractor
from .model import ActorCritic

MODEL_FILE = "ppo_model.pt"
INPUT_DIM = len(ACTIONS) * len(FEATURE_NAMES)


def vectorize_state(extractor, game_state):
    return extractor.action_features(game_state).astype(np.float32).reshape(-1)


def setup(self):
    self.device = torch.device("cuda" if self.train and torch.cuda.is_available() else "cpu")
    self.feature_extractor = FeatureExtractor()
    self.model = ActorCritic(INPUT_DIM, len(ACTIONS)).to(self.device)
    self.environment_steps = 0
    self.learner_updates = 0
    self.loaded_optimizer_state = None
    path = Path(MODEL_FILE)
    if path.exists():
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state"])
        self.environment_steps = int(checkpoint.get("environment_steps", 0))
        self.learner_updates = int(checkpoint.get("learner_updates", 0))
        self.loaded_optimizer_state = checkpoint.get("optimizer_state")
    self.model.eval()
    self.pending_action = None


def act(self, game_state):
    obs = vectorize_state(self.feature_extractor, game_state)
    mask = self.feature_extractor.safe_action_mask(game_state)
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
    with torch.no_grad():
        dist, value = self.model.distribution_and_value(obs_t, mask_t)
        action_t = dist.sample() if self.train else torch.argmax(dist.logits, dim=-1)
        log_prob = dist.log_prob(action_t)
    action_index = int(action_t.item())
    if self.train:
        self.pending_action = (
            obs.copy(), action_index, float(log_prob.item()), float(value.item()), mask.copy()
        )
    return ACTIONS[action_index]


def state_to_features(game_state):
    return None if game_state is None else vectorize_state(FeatureExtractor(), game_state)
