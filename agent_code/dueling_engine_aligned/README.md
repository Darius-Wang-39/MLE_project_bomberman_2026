# Adopted engine-aligned inference agent (2026-09-16)

Frozen copy of `dueling_hunt_center_04`; identical network and checkpoint.
No additional training. Default center Q bias remains 0.04 and objective Q bias
remains 1.00. Adopted as the default watch/package target with user approval
after the initial evaluation. Historical baselines and old ZIPs are retained;
the new package is `submission/dueling_engine_aligned.zip`.

Only explosion timing predictions change:

- No chain reactions: each bomb follows its own timer, as in `update_bombs`.
- Existing flames last `explosion_map` more decision steps, not map + 1.
- A bomb tile stays blocked during the movement phase of its detonation turn.

This is not a complete world simulator: crates and opponent positions remain
static in path search. Use only for inference; no training callbacks are bundled.
The 2x2 experiment in `evaluation/engine_timing_ablation.py` also sets the
objective Q bias to zero in memory, without deleting the learned objective
feature or changing the checkpoint.

The 160-game initial ablation gave this candidate 4.35 points/game versus the
baseline's 4.40 against three rule-based agents, and 3.25 versus 2.35 against
the fixed mixed lineup. Mixed suicide events remained 0.15/game; rule3 rose
from 0.15 to 0.20. Keep objective bias 1.00. Adoption corrects a known engine
mismatch; it is not proof of general tournament superiority or a suicide fix.
Full raw data and limitations: `experiments/records/engine_timing_ablation_20260916/`.

Code annotations: search `ENGINE-TIMING-FIX` in `features.py` for all three fixes.
Official framework checked: `ukoethe/bomberman_rl` commit `0f55c1d`.
See the 2026-09-16 teammate handoff in the root `CHANGELOG.md` before resuming
training: this directory does not include training callbacks.
