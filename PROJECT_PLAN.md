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
4. Train the DDQN curriculum and compare against uniform one-step replay.
5. Generate rule-agent demonstrations and test behaviour-cloning/replay warm
   starts.
6. Train the bomb-and-escape curriculum against `peaceful_agent` and
   `coin_collector_agent`.
7. Add an opponent checkpoint pool and self-play against frozen snapshots.
8. Tune on training seeds and evaluate on a locked seed suite against all
   supplied agents.
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
