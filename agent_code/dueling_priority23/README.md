# P2/P3 experimental continuation — 2026-09-16

Completed experiment and teammate handoff (2026-09-17): see
`experiments/records/priority23_20260917/REPORT.md` and the root `CHANGELOG.md`.
Both training checkpoints are archived with that report. **Default settings in
this directory enable both changes; they are not the validation-winning P3-only
configuration.** Use the experiment runner with `--arm p3` to reproduce that
configuration. The adopted `dueling_engine_aligned` agent remains unchanged.

This is an experimental, trainable copy of `dueling_engine_aligned`, not the
adopted watch/submission agent. Its checked-in starting checkpoint is unchanged
from that agent. Training runs in an isolated arena, not over the original ZIP.

## Changes and boundaries

- **PRIORITY-2:** Crates already covered by existing bombs are excluded from a
  new bomb's crate yield and productive target set. The existing bomb-quality
  reward consequently no longer counts those crates as new productive value.
  During escape, equally rated safe actions can be distinguished by their
  reachable endpoints' access to currently visible coins after the explosions.
  Hidden coins are never read. This does not promise optimal coin collection.
- **PRIORITY-3:** A possible same-turn collision is distinguished from a clear
  escape. When a failed move would immediately be fatal, prefer reliable safe
  alternatives, but retain a last-chance move if every option is risky. A bomb
  is rejected when its static escape routes all disappear in a stress test of
  narrow tiles reachable by a nearby opponent. Opponent reach is capped at
  three moves; distant opponents do not ban all one-corridor bombs. This is a
  conservative risk estimate, not opponent-policy prediction or a guarantee.
- Four action-relative features are appended (22 → 26): redundant bomb
  fraction, post-escape visible-coin access, contested movement, and remaining
  bomb escape space under the narrow-exit stress test. New input weights start
  at zero; old learned weights and hidden layers are preserved on migration.
- Objective Q bias stays 1.00, center Q bias 0.04 and suicide multiplier 2.5.
  Reward constants are unchanged, but target-dependent shaping and bomb-quality
  rewards now use the revised features. Optimizer and replay are reset once for
  the new representation, then retained between training opponent blocks.

`features.py` exposes `enable_economy` and `enable_contest` for frozen ablations.
When both are disabled, the original 22 feature columns and safety mask match
the adopted baseline, and the four appended columns are zero. Geometry fixes
remain in `base_features.py` for every arm. There are no additional hard-coded
Q biases for the four new features.

## Experiment

Use `evaluation/priority23_experiment.py` to prepare a new isolated arena, run
baseline/P2/P3/both frozen comparisons, and continue the both-enabled learner
for at most 200 rounds. Every 100 rounds a checkpoint is saved. Training seeds,
validation seeds and final holdout seeds are disjoint and recorded in the
arena's `protocol.json`. The same initial boards and spawn positions are used
for each evaluated candidate. Only the learner changes during training.

The source directory's checkpoint is the **starting** checkpoint. Do not confuse
the frozen logic ablation with trained checkpoints saved under `checkpoints/`.
Do not overwrite the adopted agent until independent results justify doing so.

```text
python -m unittest discover -s tests
python evaluation/priority23_experiment.py prepare --arena <new-arena> --senior-root <senior-checkout>
python evaluation/priority23_experiment.py eval --arena <arena> --arm baseline --tag baseline --rounds 10
python evaluation/priority23_experiment.py eval --arena <arena> --arm p2 --tag p2 --rounds 10
python evaluation/priority23_experiment.py eval --arena <arena> --arm p3 --tag p3 --rounds 10
python evaluation/priority23_experiment.py eval --arena <arena> --arm both --tag both --rounds 10
python evaluation/priority23_experiment.py train --arena <arena> --rounds 200
python evaluation/priority23_experiment.py eval --arena <arena> --arm both --tag trained100 --checkpoint <arena>/checkpoints/trained_0100.pt --rounds 10
python evaluation/priority23_experiment.py eval --arena <arena> --arm both --tag trained200 --checkpoint <arena>/checkpoints/trained_0200.pt --rounds 10
```

The historical trainer is retained except for importing this candidate's
callbacks/features. Typical remaining limitations: static crates/opponents in
the main survival search, no adversarial opponent search, and unchanged
emergency fallback. Read the experiment report before deciding what to adopt.
