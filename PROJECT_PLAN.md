# MLE Bomberman Project Plan

## Chosen models

### 1. `linear_q_agent`

A fast, interpretable linear Q-learning baseline. It uses one shared weight vector
for all movement directions and action-relative features. This makes rotations
and reflections equivalent by construction instead of storing duplicate replay
items.

The feature extractor exposes measurements such as legality, future escape
space, danger reduction, distance progress, and potential bomb yield. It does
not emit a hand-written "best action" label.

### 2. `dueling_ddqn_agent`

The competition candidate. It uses a small Dueling Double DQN with five-step
returns, prioritized replay, symmetry augmentation at sampling time, and leaves
room for behaviour-cloning warm-start data from the supplied `rule_based_agent`.

## Daily milestones

1. Framework, reproducible project structure, safety/feature extraction, tests,
   and a runnable linear Q-learning loop.
2. Train and evaluate the linear agent on `coin-heaven` and `classic` without
   opponents. Run symmetry and reward-shaping ablations. Task 1 is solved on the
   first held-out seed. A one-step failure and successful five-step correction
   are recorded for Task 2; multi-seed confirmation remains.
3. Implement the Dueling Double DQN, five-step prioritized replay, Double-DQN
   target updates, symmetry augmentation, and unit tests. The complete callback,
   checkpoint, and reload path has passed a three-round smoke training run. A
   subsequent 30-round `coin-heaven` curriculum collected all 50 coins in every
   round of a 20-round locked-seed evaluation with zero invalid actions.
4. Train the DDQN curriculum and compare against uniform one-step replay. The
   first 30-round single-agent `classic` stage averaged 119.5 destroyed crates
   and 8.65 coins on a 20-round held-out seed, with zero suicides and invalid
   actions; opponent pressure and round efficiency remain untested.
5. Generate rule-agent demonstrations and test behaviour-cloning/replay warm
   starts. Completed with 11,627 transitions from 50 four-player rounds and a
   round-disjoint validation split. Behaviour cloning reached 69.2% held-out
   teacher-action accuracy, but its 2.9 mean match score trailed the RL
   checkpoint's 5.25, so it is retained as an initialization rather than a
   final policy.
6. Train the bomb-and-escape curriculum against `peaceful_agent` and
   `coin_collector_agent`. Completed with 50 rounds per stage and 20 held-out
   evaluation rounds. The peaceful stage reached 11.7 score and 1.3 kills per
   round. The coin-collector stage reduced suicides to 0.05 but collapsed to
   zero kills, so both checkpoints are retained for the Day 7 opponent pool.
7. Add an opponent checkpoint pool and self-play against frozen snapshots.
   Completed with Day 4 aggressive, Day 6 high-scoring, and Day 6 safe frozen
   policies. A 50-round self-play stage improved score from 2.6 to 3.15 and
   kills from 0.05 to 0.15 on an identical 20-round ablation lineup, while
   suicides remained equal and invalid actions fell from 3.45 to 0.4.
8. Tune on training seeds and evaluate on a locked seed suite against all
   supplied agents. Completed over six fixed lineups and 120 rounds: 5.38 mean
   score, 0.317 kills, 0.267 suicides, 0.175 invalid actions per round, and 2.72
   ms mean action time. Mixed and rule-based lineups have the highest suicide
   rates and remain the primary weaknesses.
9. Run ablations, timing tests, and long tournament-style matches.
10. Freeze candidates, run Docker/pre-submission checks, and package only the
    selected agent directory.

## Evaluation rules

- Never choose a final model from a single game.
- Keep training and evaluation seeds separate.
- Report mean score, survival rate, kills, suicides, coins, invalid actions, and
  action latency.
- Preserve failed experiments and configuration files for the report.
- Keep all tournament inference single-process and below the 0.5 second limit.
