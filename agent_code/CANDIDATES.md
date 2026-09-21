# Candidate Agents

This file identifies the project's key candidate agents without adding special
characters to their directory names. The agent directories keep valid Python
module names so that command-line runs, notebooks, and imports remain portable.

## Current ranking

| Level | Agent | Model source | Key experimental result | Conclusion |
| --- | --- | --- | --- | --- |
| Selected | `dueling_hunt_center_04` | Frozen inference ablation of `dueling_ddqn_agent`; same checkpoint and safety system, with hunt-center Q bias changed to `0.04` | Best internal result in notebook 04: 3.30 score/round, 0.367 kills, 0.167 suicides, and 2.267 invalid actions over 30 rounds. Formal notebook 09 result: 2.820 score/round over 150 rounds. | Final submission recommendation based on the broader locked evaluation evidence. Do not train this directory. |
| Experimental candidate | `frozen_engine_aligned_0200` | Engine-aligned DDQN continued for 200 rounds under the notebook 10 eight-block curriculum; P2/P3 disabled | Development: 4.417 versus 3.550 for checkpoint 0000 over 60 paired games. Fresh holdout: 3.100 versus 2.893 over 150 paired games, difference +0.207 with 95% interval `[-0.247, 0.667]`. | Best checkpoint within the engine-aligned continuation experiment, but not promoted to the formal submission selection. Do not train or overwrite this directory. |
| Finalist | `dueling_ddqn_agent` | Day 18 target-curriculum Dueling Double DQN checkpoint with prioritized replay, five-step returns, safety masks, and target guidance | Agent README result: 3.62 score/round, 0.20 kills, and 0.20 suicides over 50 held-out rule-agent rounds. Formal notebook 09 result: the highest point estimate, 2.907 score/round, but only +0.087 over the selected agent with 95% paired seed-cluster CI `[-0.513, 0.660]`. | Main challenger. The formal point-estimate lead is too uncertain to overturn the broader evidence supporting `dueling_hunt_center_04`. |
| Reserve | `hunt_center_04_mix_0800` | Checkpoint after 800 rounds of controlled mixed-opponent continuation from `dueling_hunt_center_04`, produced by notebook 08 | Final holdout: 3.583 score/round, 0.308 win rate, 0.150 suicides, and 5.150 invalid actions over 120 rounds. Notebook 09 again found a high 8.407 invalid actions/round. | Retain for continuation-study reproducibility, but do not promote for submission because the development gain did not replicate and invalid actions remained high. |
| Baseline | `frozen_sarsa_best` | Frozen stage-2 linear SARSA(lambda) checkpoint: lambda `0.4`, final alpha `0.001`, 153 training episodes | Final selection result: 3.8 score/round, 0.45 win rate, 0.35 suicides/round, and 0.65 survival over 20 rounds. In the notebook 03 comparison, SARSA scored 3.020 against rule agents and 3.740 in mixed games. | Keep as the strongest frozen SARSA baseline and as an algorithmically distinct comparison; it is not the submission agent. |
| Baseline / stop-line | `linear_q_final` | Final controlled linear-Q experiment, warm-started from the 61,973-update baseline and expanded with eight Task-4 interaction features | The best experimental branch (`alpha=0.003`, `n_step=8`) reached 3.32 in validation; locked evaluation scored 2.98 versus 3.02 for the original baseline while improving suicides from 0.43 to 0.29 and survival from 34% to 62%. | Keep as a reproducible stop-line experiment. It improved safety but did not satisfy the promotion gates. The selected frozen checkpoint is stored separately as `frozen_linear_q_final`. |

## Evidence and maintenance

- `notebooks/03_compare_task4_candidates.ipynb`: frozen candidate comparison.
- `notebooks/04_internal_vs_external_benchmarks.ipynb`: internal agents versus
  fixed external benchmarks.
- `notebooks/05_sarsa_lambda_tasks1_4.ipynb`: SARSA(lambda) training and
  selection evidence.
- `notebooks/06_linear_q_final_stopline.ipynb`: controlled linear-Q stop-line
  experiment.
- `notebooks/08_continue_training_dddqn.ipynb`: controlled DDQN continuation
  and one-shot final holdout.
- `notebooks/09_formal_mixed_benchmark.ipynb`: final paired mixed-opponent
  benchmark.
- `notebooks/10_continue_training_engine_aligned_dddqn.ipynb`: engine-aligned
  continuation, invalid-action diagnosis, checkpoint selection and paired
  holdout.
- `CHANGELOG.md`: chronological experiment descriptions and decisions.

Update this table when a candidate is promoted, rejected, retrained, or replaced.
Keep frozen agents inference-only, and record new evidence before changing the
final recommendation.
