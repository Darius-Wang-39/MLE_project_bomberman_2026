# SARSA(lambda) agent

Linear on-policy SARSA with accumulating eligibility traces, the repository's
17-dimensional action-relative features, potential shaping, and safe-action
mask. The transition is cached by `game_events_occurred` and updated only after
the next behaviour action is actually selected by `act`.

```bash
.venv/bin/python main.py play --no-gui --agents sarsa-lambda --train 1 --scenario coin-heaven --n-rounds 200 --seed 51001
```
