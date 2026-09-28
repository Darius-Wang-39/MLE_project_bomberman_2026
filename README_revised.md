# MLE 2026 Bomberman

Team repository for the Heidelberg MLE project and competition, based on the
official `bomberman_rl` framework.

All team members should record meaningful updates in
[CHANGELOG.md](CHANGELOG.md) using the date/change/method/result format before
pushing or merging them.

## Final paper and submission snapshot

The final snapshot reported in the paper is **AegisHunter-04**, implemented by
`dueling_hunt_center_04`. It is a frozen, inference-only copy of the selected
Day 18 checkpoint. It uses the safety-aware action filter, objective Q bias of
1.00, and hunt-center bias of 0.04. This is the version to treat as the final
paper/submission agent.

`dueling_engine_aligned` is a later inference-time engine-alignment branch. It
retains the learned weights but changes three timing/prediction rules to match
the official game engine. Those changes were made after the frozen AegisHunter-04
snapshot and are not part of the final version reported in the paper. Do not use
`dueling_engine_aligned` as the final paper/submission version.

`dueling_priority23` is a separate experimental branch for resource-efficiency
and contested-exit ideas. Its continuation checkpoints and validation/holdout
comparisons are documented in
[the P2/P3 report](experiments/records/priority23_20260917/REPORT.md). These
experiments are outside the paper's final AegisHunter-04 snapshot and are not
submission checkpoints.

## Other agents and project notes

- `linear_q_agent`: fast, interpretable linear Q-learning baseline with exact
  D4 symmetry through shared action-relative features and five-step returns.
- `dueling_ddqn_agent`: compact Dueling Double DQN training agent with five-step
  returns, prioritized replay, Double-DQN targets, safe-action masking, and D4
  replay augmentation.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the training schedule and evaluation
rules, and [experiments/README.md](experiments/README.md) for experiment logging.
The dated entries in `CHANGELOG.md` record repository work as it happened; they
should not override the final version identified by the paper. In particular,
the 2026-09-16 entry describes the later `dueling_engine_aligned` branch.

## Quick checks

```bash
python -m unittest discover -s tests -v
python main.py play --no-gui --agents linear_q_agent --scenario coin-heaven --n-rounds 10
python main.py play --no-gui --agents dueling_ddqn_agent --scenario coin-heaven --n-rounds 10
```

Training is enabled with `--train 1`. Keep training and evaluation seeds
separate; raw framework output under `results/` stays local. These training
examples apply to trainable agents, not to the frozen `dueling_hunt_center_04`
submission snapshot or the inference-only `dueling_engine_aligned` branch.

## Watch and package the final snapshot

To watch AegisHunter-04 in one seeded classic round against three supplied
rule-based agents, run:

```bash
python main.py play --agents dueling_hunt_center_04 rule_based_agent rule_based_agent rule_based_agent --scenario classic --n-rounds 1 --seed 240824 --update-interval 0.15 --save-replay replays/aegishunter04_demo.pt --match-name aegishunter04-demo
```

The framework assigns player colors by popping the configured agent list. The
GUI requires `pygame`; install it with `python -m pip install pygame` if needed.
The replay is saved locally at `replays/aegishunter04_demo.pt`.

Package the final snapshot from its own agent directory:

```bash
python tools/package_submission.py agent_code/dueling_hunt_center_04 --output submission/dueling_hunt_center_04.zip --manifest submission/dueling_hunt_center_04_manifest.json
```

Do not substitute the `dueling_engine_aligned` archive when reproducing the
paper's final snapshot. The CHANGELOG records that no archive had been uploaded
to MaMPF as of 2026-09-17; check the competition-platform status separately.
