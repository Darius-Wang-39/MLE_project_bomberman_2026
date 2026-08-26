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

## 2026-08-26

### Day 15 — set future suicide multiplier to 2.5

- **Change:** Changed the requested future-training suicide multiplier from
  `3.0` to `2.5`, so `KILLED_SELF` now contributes `-50` instead of `-60`.
- **Method:** Updated the single reward constant, its locking unit test, agent
  documentation, and a staged-training plan. Historical Day 14 records remain
  unchanged. The active checkpoint is explicitly documented as still trained
  with `3.0`; reward-code changes affect behaviour only after retraining.
- **Result:** Future training is configured for `2.5`. The recommended next run
  is at most 300 rounds in 50-round chunks, with a separate 100-round held-out
  evaluation after every chunk and promotion of the best checkpoint rather
  than automatically keeping the last one. All 21 tests pass. The rebuilt
  283,498-byte submission archive passed CRC and syntax checks and has SHA-256
  `48C745AF5CE21D4E524362D0F2457F1A54C5D9BBD5E089E6EFE9417DB7A41B4F`.
- **Commit:** this commit

## 2026-08-24

### Day 14 — 3.0 suicide-penalty multiplier and visual match

- **Change:** Set the requested suicide-penalty multiplier to `3.0`, changing
  the `KILLED_SELF` event reward from `-20` to `-60`, retrained the promoted
  tactical model, and launched a visible match against three rule-based agents.
- **Method:** Preserved the exact Day 13 checkpoint, ran 100 rule-agent safety
  rounds plus 50 mixed-retention rounds, then compared old and new weights over
  100 zero-exploration rounds on seed `8701`. `GOT_KILLED` remains `-15`, so the
  stronger multiplier specifically distinguishes self-kills. A unit test locks
  the multiplier and resulting event reward.
- **Result:** The candidate scored 392 versus the baseline's 361 (+8.6%) and
  beat all three rule agents (281, 280, 266). Bomb use fell from 29.72 to 24.43
  per round and kills rose from 0.10 to 0.14; suicides did not improve, moving
  from 0.51 to 0.56, and invalid actions rose from 0.90 to 1.10. The requested
  candidate was promoted with this tradeoff documented. All 21 tests pass. A
  visible seed-8801 match identifies the promoted agent as pink and saves its
  replay under `replays/day14_suicide3_visual.pt`. The promoted checkpoint
  SHA-256 is
  `C1AF4217707AEE90677CDDD459DBBAA65C9B7C00991EEA7F562EF4400F746492`.
  The rebuilt 283,401-byte submission ZIP passed CRC and syntax checks and has
  SHA-256
  `E2C247A0667D7BFAD2C67E8080684C0012F984049E52E9D16F2A75A697A1223C`.
- **Commit:** this commit

### Day 13 — rule-based tactical specialization

- **Change:** Diagnosed a fresh 100-round three-rule baseline, appended tactical
  bomb-pressure and armed-opponent-threat features, added a contextual penalty
  for bombs with no resource or tactical value, trained a 150-round rule-agent
  stage plus a 75-round mixed retention stage, and promoted the held-out winner.
- **Method:** The opponent-pressure feature compares each opponent's reachable
  positions at bomb detonation with and without a candidate bomb. Old 18-feature
  checkpoints migrate exactly to 20 features with only the appended columns
  zero-initialized. The `-0.25` low-value penalty applies only when a bomb has no
  crate yield, no opponent in its current blast, and less than `0.15` escape
  pressure. Promotion used a new 100-round seed against three rule-based agents,
  followed by the original six-lineup 120-round locked suite.
- **Result:** The initial baseline scored 329 versus rule-agent totals 334, 274,
  and 261. On held-out seed `8501`, the old model scored 360 and the new model
  scored 397 (+10.3%); the candidate beat all rule agents (268, 265, 244),
  raised coins from 2.75 to 3.27, reduced invalid actions from 1.06 to 0.93, and
  changed suicides from 0.61 to 0.63 per round. Across the 120-round locked
  suite, mean score rose from 6.53 to 7.77, kills from 0.40 to 0.467, and invalid
  actions fell from 0.442 to 0.333; suicides increased from 0.242 to 0.35 and
  are explicitly retained as the next weakness. All 20 tests pass and mean
  inference latency is 2.16 ms. The promoted model SHA-256 is
  `C624C7573256886FCA58C930E813AC2B0BEE34E7399F411031E3F93F2A78D2F5`.
  The deterministic 281,896-byte submission ZIP has SHA-256
  `2863004CD7EFC08D84499386C795367F5B5DE6A817CC72A77C509A5211A07060`;
  its isolated ten-round extraction test averaged 4.3 points, 0.2 kills, 0.4
  suicides, and 2.66 ms per action.
- **Commit:** this commit

### Day 12 — soft central 11x11 endgame preference

- **Change:** Added a tunable soft preference for the inclusive central box
  `x,y in [3,13]`, active only after coins and crates are gone while opponents
  remain. Frozen the promoted Day 11 checkpoint for paired comparison.
- **Method:** Appended one D4-invariant action feature measuring progress toward
  the central box and added a `0.20` center-closeness term to the endgame state
  potential. Added automatic 17-to-18 feature checkpoint migration that copies
  every old input weight into its corresponding action row and initializes only
  the six new feature columns. Escape remains controlled by the six-step safety
  mask; the center preference is not a hard movement restriction.
- **Result:** All 18 tests pass, including exact old-weight migration and center
  progress tests. The trained `0.20` potential was rejected: it raised central
  occupancy only from 28.8% to 33.4% while reducing paired mean score from 4.47
  to 3.73. Restoring the Day 11 weights and applying a `0.08` inference Q bias
  raised central occupancy to 88.2%, reduced mean box distance from 1.60 to
  0.18 tiles, and reduced suicides from 0.63 to 0.43 in the paired 30-round
  comparison. Its 120-round locked-suite mean score was 6.53 versus Day 11's
  6.41, so the lightweight bias was promoted and the training potential default
  was disabled. The resulting 262,673-byte archive has SHA-256
  `AAA79E5B65C01A5DFEB2CE278133B67EE83BCFB78F13CEBC167BEB28417E0130`;
  an isolated five-round run averaged 7.6 points with zero suicides or invalid
  actions and confirmed automatic checkpoint migration works after extraction.
- **Commit:** this commit

### Day 11 — endgame anti-stall candidate

- **Change:** Started a separate candidate from the exact Day 7 checkpoint,
  added opponent pursuit to target-free endgame shaping, strengthened the wait
  penalty, and added a safe endgame revisit penalty. The frozen submission
  checkpoint remains available as `frozen_ddqn_day7` for fair comparisons.
- **Method:** Positive closeness potentials preserve the coin-then-crate
  curriculum and activate opponent distance only when both objectives are gone.
  Revisit penalties use an eight-position training history and are disabled
  while bombs or explosions require evasive movement. Added focused unit tests
  and a two-stage, seed-separated endgame/retention curriculum configuration.
- **Result:** All 16 tests pass. Training added 31,408 environment steps and
  7,351 learner updates. In a paired 30-round evaluation, wait rate fell from
  33.9% to 3.9%, the worst wait streak fell from 216 to 7, mean score rose from
  3.77 to 4.33, and suicides were unchanged. Across the original 120-round
  locked suite, mean score rose from 5.38 to 6.41, kills from 0.317 to 0.442,
  and suicides fell from 0.267 to 0.225. Invalid actions rose from 0.175 to
  0.408 per round. The original seed-240824 visual regression fell from 212
  waits to 10, with a longest wait streak of three. The promoted 261,656-byte
  archive has SHA-256
  `DF21B0C0F74CE3058B06195BFE51599740F9E3C814F30042433BAF6E6C2FE91F`;
  its isolated five-round smoke test averaged 6.8 points with zero suicides or
  invalid actions at 1.58 ms per action.
- **Commit:** this commit

### Local Docker preparation and visual match launcher

- **Change:** Downloaded the official Docker Desktop installer to the D drive,
  added a one-click Windows launcher for watching the frozen agent, and
  documented the equivalent cross-platform GUI command.
- **Method:** Verified Windows, CPU virtualization, WSL, and disk status;
  SHA-256 checked the downloaded installer; installed only the 10.6 MB Pygame
  GUI dependency; then launched the frozen Dueling DDQN against three supplied
  rule-based agents on a fixed classic-map seed with replay saving enabled.
- **Result:** The 628.7 MB installer is stored outside the repository at
  `D:\Installers\DockerDesktop\Docker Desktop Installer.exe` with SHA-256
  `9AC03D4E900C0FDEE981D4BDE083A55FDFB28FFBA2CAE77726EFF2A437254822`.
  The native GUI opened successfully as `BombeRLe | Round #1`. Docker remains
  uninstalled until BIOS/UEFI virtualization is enabled and WSL 2 is installed.
  Replay inspection later confirmed that the first configured agent is rendered
  pink, not blue; the README identification was corrected. The frozen
  submission archive and model were not modified.
- **Commit:** this commit

### Day 10 — candidate freeze and submission package

- **Change:** Froze the Day 7 checkpoint, added deterministic packaging and
  package-exclusion tests, generated the agent-only ZIP and SHA-256 manifest,
  and ran the extracted archive under a temporary framework agent name.
- **Method:** Packaging sorts files, fixes ZIP timestamps, includes only source,
  checkpoint, and documentation suffixes, excludes logs/caches, verifies CRC,
  compiles every archived Python file, and records per-file hashes. The active
  and frozen checkpoint hashes were compared directly.
- **Result:** All 14 tests pass. The 259,433-byte archive contains eight files
  and has SHA-256 `83ab316bcb2c9667728c08a24d5bfd967f9257ce1fcb78b6b535763fb4d77c1f`.
  The extracted ZIP completed five rounds with zero suicides/invalid actions at
  1.86 ms per action. Docker was not run because the executable is absent; the
  official Dockerfile statically installs the required NumPy and PyTorch.
- **Commit:** this commit

### Day 9 — safety ablation and 100-round stress match

- **Change:** Added a legal-only inference adapter, ran a paired 30-round safety
  ablation, and ran the untouched Day 7 candidate for 100 mixed-opponent rounds.
- **Method:** The ablation kept network architecture, weights, features, lineup,
  and seed fixed; only the six-step time-expanded action mask was replaced by
  immediate legality. The long match used another locked seed and no training.
- **Result:** Legal-only committed suicide in all 30 rounds, scored zero, and
  survived just 160 total steps. The full mask scored 3.9 with 0.5 suicides per
  round. The 100-round candidate stress test completed 27,684 actions without
  error at 2.07 ms mean latency, averaging 3.87 score and 0.28 kills per round.
- **Commit:** this commit

### Day 8 — locked built-in-opponent evaluation suite

- **Change:** Added a configurable, resumable locked-suite runner and evaluated
  the Day 7 candidate for 20 rounds each against peaceful, coin-collector,
  rule-based, random, fail, and mixed lineups. Raw stats and the aggregate report
  are committed.
- **Method:** Each lineup has its own fixed seed and output file. The runner
  builds explicit framework commands, validates three opponents, resumes only
  completed outputs, and aggregates with the same metric implementation used by
  earlier experiments. Only the intentionally crashing `fail_agent` lineup uses
  `--silence-errors`.
- **Result:** Across 120 rounds the candidate averaged 5.38 score, 3.8 coins,
  0.317 kills, 0.267 suicides, and 0.175 invalid actions per round at 2.72 ms per
  action. The mixed (0.7 suicides) and rule-based (0.5) lineups are weakest. The
  first runner attempt exposed a relative-path bug; resume support fixed it. The
  fail lineup then confirmed its supplied agent deliberately raises, so error
  fallback was scoped to that lineup. All 13 tests pass.
- **Commit:** this commit

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
