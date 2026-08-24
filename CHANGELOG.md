# Team Update Log

This is the single source of truth for project updates. Every meaningful code,
training, evaluation, configuration, or collaboration change must add an entry
here before it is pushed.

Each entry records:

- **Date**: `YYYY-MM-DD` in Europe/Berlin time.
- **Change**: what changed.
- **Method**: how it was implemented or tested.
- **Result**: measurable outcome or current status.
- **Commit**: the related Git commit, added after committing when practical.

## 2026-08-24

### Day 7 — frozen checkpoint pool and self-play

- **Change:** Added three independent frozen DDQN adapters for the Day 4
  aggressive, Day 6 high-scoring, and Day 6 conservative checkpoints; trained
  the live candidate against all three for 50 rounds and froze the result.
- **Method:** Each opponent loads a physically separate checkpoint and runs only
  inference. Evaluation used a 20-round full-pool seed plus a fair 20-round
  ablation where Day 6 and Day 7 faced the exact same three opponents and seed.
- **Result:** On the fair ablation, Day 7 scored 3.15 versus 2.6, achieved 0.15
  versus 0.05 kills, and reduced invalid actions from 3.45 to 0.4 while both
  candidates had 0.45 suicides per round. The self-play checkpoint is retained.
- **Commit:** this commit

### Day 6 — staged opponent curriculum

- **Change:** Trained and froze two 50-round DDQN stages against three peaceful
  opponents and then three coin collectors; evaluated each on 20 disjoint-seed
  rounds and added a combined experiment record.
- **Method:** Continued the Day 4 RL checkpoint with identical DDQN/PER settings.
  Each stage used a distinct training seed and a locked evaluation seed, and the
  intermediate checkpoint was preserved before advancing the curriculum.
- **Result:** The peaceful stage averaged 11.7 score, 1.3 kills, and 0.2
  suicides per round. The coin-collector stage reached 0.05 suicides and 0.05
  invalid actions per round but zero kills and 2.05 score, revealing an
  over-conservative policy. Both checkpoints feed the Day 7 opponent pool.
- **Commit:** this commit

### Day 5 — rule demonstrations and behaviour cloning

- **Change:** Added a development-only rule demonstration collector, a
  round-disjoint behaviour-cloning pretrainer, a separate BC evaluation adapter,
  two tests, the dataset, checkpoint, and comparison record.
- **Method:** Collected 11,627 state/action samples over 50 seeded four-player
  rounds. D4 augmentation was applied online during 15 supervised epochs; entire
  rounds, not random frames, were held out for validation. The same 20-round
  lineup and seed compared the BC policy with the existing RL checkpoint.
- **Result:** Held-out teacher-action accuracy was 69.2%. BC scored 2.9 versus
  5.25 for the RL checkpoint, so BC is useful only as a warm start. The supplied
  teacher returned `None` 185 times; the collector now counts these and applies
  an explicit safe fallback. A first CLI run also exposed and fixed a project
  import-path error. All 12 tests pass.
- **Commit:** this commit

## 2026-08-16

### Team collaboration workflow

- **Change:** Established this shared update log and a pull-request checklist;
  started onboarding teammate `pengna.sun@stud.uni-heidelberg.de` to the private
  repository.
- **Method:** Added a fixed date/change/method/result format and linked the log
  from the project README. GitHub collaborator access is handled from the
  repository access settings.
- **Result:** Future updates have one reviewable history in `CHANGELOG.md`.
  GitHub matched the email to `@Pengna-Sun`; the collaborator invitation was
  sent and is awaiting acceptance. A personal private repository grants the
  highest available collaborator level (read/write) after acceptance, while
  owner-only settings remain with `@Darius-Wang-39`.
- **Commit:** this commit

## 2026-08-15

### Dueling DDQN bomb-and-escape curriculum

- **Change:** Continued the DDQN from the coin checkpoint for 30 single-agent
  `classic` rounds and evaluated it on 20 held-out rounds.
- **Method:** Dueling Double DQN, Double-DQN targets, five-step returns,
  prioritized replay, D4 replay augmentation, potential shaping, and safe-action
  masking.
- **Result:** 119.5 crates, 8.65 coins, and 47.8 bombs per round; zero suicides
  and invalid actions; 1.59 ms mean action time.
- **Commit:** `4b56919`

### Dueling DDQN coin curriculum

- **Change:** Trained the DDQN for 30 `coin-heaven` rounds after a three-round
  smoke run and evaluated it on 20 held-out rounds.
- **Method:** Used the same action-relative feature representation as the linear
  agent with prioritized replay and random D4 augmentation at sample time.
- **Result:** Collected all 50 coins in all 20 evaluation rounds; zero suicides
  and invalid actions; 2.57 ms mean action time.
- **Commit:** `bee8b02`

### Initial two-agent implementation

- **Change:** Added `linear_q_agent`, `dueling_ddqn_agent`, shared safety
  features, symmetry utilities, evaluation summarization, experiment records,
  tests, and the daily project plan.
- **Method:** The linear baseline uses shared action-relative weights and
  five-step Q-learning. The competition candidate uses a compact dueling neural
  network with prioritized n-step replay.
- **Result:** Ten unit tests passed. The linear agent solved `coin-heaven` and
  learned bomb-and-escape behaviour in `classic` after replacing one-step credit
  assignment with five-step returns.
- **Commit:** `8324e9d`

## Entry template

```markdown
## YYYY-MM-DD

### Short update title

- **Change:**
- **Method:**
- **Result:**
- **Commit:**
```
