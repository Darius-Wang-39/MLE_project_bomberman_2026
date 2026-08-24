"""Training hooks that flush the offline demonstration dataset each round."""

from __future__ import annotations

from .callbacks import save_demonstrations


def setup_training(self):
    pass


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    pass


def end_of_round(self, last_game_state, last_action, events):
    save_demonstrations(self)

