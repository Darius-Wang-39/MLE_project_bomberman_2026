# Experiments

Every run should record:

- git commit and agent configuration;
- training and evaluation seeds;
- opponent lineup and scenario;
- number of rounds;
- mean score, coins, kills, suicides, survival, invalid actions, and latency.

Training seeds and locked evaluation seeds must not overlap. Raw result files are
kept locally under `results/`; concise aggregate tables and plotting code should
be committed for report reproducibility.

## Controlled five-family experiment

`evaluation/run_controlled_family_24h.py` runs the resumable Tier 1--3 protocol
defined in `experiments/configs/controlled_family_24h.json`. It compares controlled
Linear Q, SARSA(lambda), PPO-no-D4, DDDQN-no-D4 and DDDQN-D4 learners from scratch.
Use `evaluation/summarize_controlled_family_24h.py` after Tier 3 to regenerate the
JSON/CSV summaries, report tables, and figures in `results/controlled_family_24h/`.
