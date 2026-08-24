# MLE 2026 Bomberman

Team repository for the Heidelberg MLE project and competition, based on the
official `bomberman_rl` framework.

All team members must record meaningful updates in
[CHANGELOG.md](CHANGELOG.md) using the date/change/method/result format before
pushing or merging them.

## Agents

- `linear_q_agent`: fast, interpretable linear Q-learning baseline with exact
  D4 symmetry through shared action-relative features and five-step returns.
- `dueling_ddqn_agent`: compact Dueling Double DQN with five-step returns,
  prioritized replay, Double-DQN targets, safe-action masking, and D4 replay
  augmentation.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the training schedule and evaluation
rules, and [experiments/README.md](experiments/README.md) for experiment logging.

## Quick checks

```bash
python -m unittest discover -s tests -v
python main.py play --no-gui --agents linear_q_agent --scenario coin-heaven --n-rounds 10
python main.py play --no-gui --agents dueling_ddqn_agent --scenario coin-heaven --n-rounds 10
```

Training is enabled with `--train 1`. Keep training and evaluation seeds
separate; raw framework output under `results/` stays local.

The frozen competition archive and its per-file checksum manifest are under
[`submission/`](submission/). Rebuild them with
`python tools/package_submission.py agent_code/dueling_ddqn_agent --output
submission/dueling_ddqn_agent.zip --manifest
submission/dueling_ddqn_agent_manifest.json`.

