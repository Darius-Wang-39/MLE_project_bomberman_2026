# MLE 2026 Bomberman

Team repository for the Heidelberg MLE project and competition, based on the
official `bomberman_rl` framework.

All team members must record meaningful updates in
[CHANGELOG.md](CHANGELOG.md) using the date/change/method/result format before
pushing or merging them.

## Agents

- **Current adopted inference agent (2026-09-16): `dueling_engine_aligned`.**
  Corrects three prediction/official-engine mismatches, with unchanged weights
  from `dueling_hunt_center_04`, objective bias 1.00 and center bias 0.04.
  This is the default watch/package target, not a newly trained model.
  See the dated Chinese teammate handoff in [CHANGELOG.md](CHANGELOG.md).
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

## Watch the frozen agent

On Windows, double-click `watch_agent.cmd` to watch the frozen Dueling DDQN
play one seeded classic round against three supplied rule-based agents. The pink
player is `dueling_engine_aligned` (the framework assigns colors by popping the
configured list). The match is also saved locally as
`replays/visual_demo.pt` for replay. The GUI requires `pygame`; install it with
`python -m pip install pygame` if the launcher reports that it is missing.

The equivalent cross-platform command is:

```bash
python main.py play --agents dueling_engine_aligned rule_based_agent rule_based_agent rule_based_agent --scenario classic --n-rounds 1 --seed 240824 --update-interval 0.15 --save-replay replays/visual_demo.pt --match-name visual-demo
```

The current agent archive is `submission/dueling_engine_aligned.zip`; older
archives are retained for comparison and are not the adopted version. Rebuild:

```bash
python tools/package_submission.py agent_code/dueling_engine_aligned --output submission/dueling_engine_aligned.zip --manifest submission/dueling_engine_aligned_manifest.json
```

This directory is inference-only: do not pass `--train 1` for it. The quick-check
training examples above refer to older trainable agents, not this adopted
inference snapshot. Before continuing training, explicitly port the same three
timing fixes into the intended training agent and test it; historical directories
were deliberately not rewritten. No archive has been uploaded to MaMPF.

