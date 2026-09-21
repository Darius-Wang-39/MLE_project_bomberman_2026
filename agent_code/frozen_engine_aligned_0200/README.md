# Frozen engine-aligned DDQN — continuation checkpoint 0200

This inference-only agent freezes `engine_aligned_adapt_0200.pt`, selected after
the paired development evaluation in notebook 10.  It uses the unchanged
engine-aligned 22-feature inference code: P2 and P3 remain disabled, objective
Q bias is `1.00`, and center Q bias is `0.04`.

The checkpoint completed 200 continuation-training rounds.  On the 60-game
development set it scored 4.417 points per round versus 3.550 for checkpoint
0000.  Its invalid movements are retained intentionally: deterministic replay
showed that all were same-turn target contention after another agent moved
first, rather than illegal decisions in the observed state.

On the subsequent locked 150-game paired holdout it scored 3.100 versus 2.893
for checkpoint 0000.  The paired difference was +0.207 with bootstrap 95%
interval `[-0.247, 0.667]`: positive, but not statistically decisive.

Do not train or overwrite this directory.  The canonical frozen checkpoint and
selection metadata live in
`experiments/runs/engine_aligned_adapt_20260917/best/`.
