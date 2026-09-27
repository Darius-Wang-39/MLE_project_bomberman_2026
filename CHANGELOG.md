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

## 2026-09-27

### Five-family equal-budget comparison and D4 ablation

- **Change:** Added isolated controlled Linear Q, SARSA(lambda), PPO, DDDQN-no-D4,
  and DDDQN-D4 agents plus a resumable three-tier training/evaluation runner.
- **Method:** All five learners started from scratch and shared the original
  17 action-relative features, safety mask, event reward, shaping, curriculum,
  lineups and evaluation seeds. Three training seeds each received a 380-round
  foundation curriculum and independent 25/50/100-round rule-agent branches.
  The DDDQN pair used identical initial conditions and separate exploration,
  replay and augmentation RNG streams; only replay-time D4 differed.
- **Result:** At the equal-budget anchor, DDDQN-D4 led Task 2/3/4 with 111.77
  crates, 2.07 kills and 3.67 score per round; DDDQN-no-D4 was second at
  105.40/1.33/3.40. Naive 100-round rule continuation regressed both DDDQNs,
  revealing forgetting rather than monotonic headroom. On the held-out mixed
  lineup, their B100 scores were 5.05/4.10 versus at most 2.85 for the other
  families. Mean action latency remained about 1 ms. Full results and the
  report-ready interpretation are under `results/controlled_family_24h/`.
- **Validation:** Completed 8,325 training and 2,400 evaluation rounds. Checked
  all 90 training records, 165 evaluation records and 90 checkpoint hashes;
  32 existing learner tests plus three controlled-protocol/D4 checks passed.
- **Decision:** Retain DDDQN as the main development platform because it learned
  the strongest shared-budget multi-task policy at modest absolute cost. Treat
  D4 as a useful early-data aid, not a guarantee of stable long continuation;
  future training should add rehearsal/retention and multi-task checkpoint gates.
- **Commit:** pending

## 2026-09-17

### 预跑前最小修复：DDQN 回合末经验去重，不更换比赛模型

- **改动 / Change：** 修复 `dueling_ddqn_agent/train.py` 和当前续训实验入口
  `dueling_priority23/train.py` 的存活回合末动作重复入池问题；搜索
  `TERMINAL-RECORD-FIX` 可定位。历史 `dueling_hunt_center_04_mix_train`
  等旧实验目录不在本次修复范围，不应据此认为所有历史 trainer 均已修复。
- **办法 / Method：** 保留一个待确认的普通回调，下一次回调到来后才提交。
  如果结束回调对应同一步，将它作为一次终止转移提交，结束事件只计算一次；
  如果是死亡的新一步，先提交此前待确认的普通转移，再记录死亡动作。
  事件列表复制保存，防止引擎追加存活事件污染先前经验。
  n-step 序列、步计数和优化调用均按真实动作处理，回合末清空缓存。
  普通经验进入训练延迟一个回调，不修改引擎、网络结构或任何奖励常数。
- **验证 / Validation：** 新增 8 项回归测试，每项覆盖两个训练模块，包含
  老师本地引擎实际 `send_game_events/end_round` 回调顺序、400 步存活结束、
  首步死亡、后续死亡、5 步回报边界、事件列表变化和跨回合清理。
  全套 69 项测试通过；先前“一个动作生成两条经验”的复现现在只有一条。
- **预跑冻结 / Freeze：** 没有重新训练，没有调整目标奖励、抢格策略、
  手工 Q 偏置或安全过滤。当前采用的 `dueling_engine_aligned` 整个目录、
  `watch_agent.cmd` 和 `submission/dueling_engine_aligned.zip` 均未改变。
  权重 SHA256 仍为 `709cca4351f3bc5893f766b01b8c9335fa299d25331a275ae281b123df31009a`；
  提交包 SHA256 仍为 `bceeaa102109915c285ffd27174d16c58693dc7d580237e9dbff7627ea8303a4`。
- **交接 / Handoff：** 本修复只防止今后续训重复记账，不会修复旧权重中的
  既有学习结果，也不会立即改变预跑表现。旧训练记录和计数不追溯改写。
  抢格、失败动作仍有目标加分、行为偏置与学习目标的调整留待预跑后单独验证。
  预跑仍使用现用引擎修正版，不使用 P2/P3 的 100/200 局实验权重。

### 优先级 2 / 3 实验实现、200 局续训及交接

- **日期 / Date：** 9 月 16 日开始修改和训练，9 月 17 日完成比较。
- **改动 / Change：** 新增独立实验目录 `agent_code/dueling_priority23`。
  第 2 项扣除已被场上炸弹覆盖的箱子收益，并在逃生质量相同的选择中考虑
  爆炸后到可见金币的距离；不使用隐藏金币信息。第 3 项识别同回合抢位风险，
  并对附近对手可达的狭窄出口做压力测试；不一概封禁单通道，保留最后的逃生机会。
  代码中搜索 `PRIORITY-2` / `PRIORITY-3` 可定位。两项可分别开关。
- **办法 / Method：** 保留第 1 项引擎修复、目标 Q 加分 1.0、中心 Q 加分 0.04、
  自杀系数 2.5。动作特征由 22 列扩展到 26 列，新增输入权重从零开始，
  继承原网络权重；新增特征在冻结对照中不会凭空产生已学贡献。
  训练时重置一次优化器和经验池，之后在 25 局一个对手区段之间持续保留。
  奖励常数不变，但目标距离、炸弹收益相关奖励按修正后的特征计算。
- **训练 / Training：** 两项都开启，完成 200 局混合对手续训，实际约 25 分 32 秒。
  引擎记录 63,329 次动作，增加 15,615 次网络更新；训练步计数增加 63,465，
  包含存活局的终止回调，不能等同于不同的游戏动作。保存了 100/200 局权重。
  原采用版本的权重、观看入口和提交 ZIP 均未覆盖。
- **验证 / Validation：** 61 项单元测试通过。6 个候选各用 10 局规则组和
  10 局学长混合组比较，共 120 局；种子与训练分开、初始地图及出生位置配对。
  规则组 / 混合组均分：基线 3.80 / 2.20；P2 4.50 / 1.50；P3 4.00 / 2.90；
  两项开启未续训 3.90 / 2.70；训练 100 局 2.70 / 1.50；训练 200 局 3.30 / 2.00。
  按预先约定的两组平均得分，选出 P3-only，随后与基线在另一批种子各测 40 局，
  该独立测试不用于重新挑选其他版本。完整结果和限制见下方报告。
- **独立测试 / Holdout：** P3-only 对规则组均分 3.95，基线 3.75；混合组
  两者均为 2.55。P3 的混合组自杀记录 4/20 局，基线 2/20；无效动作
  4.45 / 3.85 次每局，第一名率（含并列）20% / 25%。配对均分差 95%
  描述性 bootstrap 区间：规则组 [-0.60, 1.15]，混合组 [-0.75, 0.75]。
  验证集的优势未在独立强对手组得到确认，因此不宣称稳定升级。
- **决定 / Decision：** 第 2、3 项属于待验证的策略改进，不是已确定的引擎错误。
  不因完成训练而自动替换当前正式候选；当前仍使用 `dueling_engine_aligned`。
  训练权重没有在验证集中胜出，不建议直接作为提交版本。
- **交接 / Handoff：** 完整代码、原始成绩、选型记录及两份训练权重位于
  `experiments/records/priority23_20260917/`。源码实验目录中的 `.pt` 是训练起点，
  不要误认成 200 局版。默认开启两项；复现 P3-only 应使用脚本 `--arm p3`。
  `evaluation/priority23_experiment.py` 负责隔离训练/比赛，
  `evaluation/summarize_priority23.py` 核对配对条件、汇总并归档。
  没有发送消息给队友，也没有上传 MaMPF；请阅读 `REPORT.md` 后决定是否另行采用。

## 2026-09-16

### 采用引擎预测修正版 — 给队友的交接说明

- **日期 / Date：** 2026-09-16。用户确认采用前一条实验记录中的预测修复。
- **改动 / Change：** 当前默认观看、打包目录改为
  `agent_code/dueling_engine_aligned`，`watch_agent.cmd` 和首页说明同步更新。
  在 `features.py` 三处添加 `ENGINE-TIMING-FIX 2026-09-16` 标记，便于搜索定位。
  当前压缩包为 `submission/dueling_engine_aligned.zip`；旧目录和旧压缩包保留，
  不要误将旧版当成修正版。未上传 MaMPF，也未直接联系队友。
- **原因 / Why：** 项目说明第 2 页规定正式比赛使用老师的原始框架，第 3 页
  指定 `ukoethe/bomberman_rl`。重新获取官方仓库后核对提交 `0f55c1d`：本地
  移动、炸弹更新、火焰更新、伤害判定、状态输出与官方关键函数一致；本地
  `environment.py` 的差别仅是额外成绩统计。此次修的是 agent 的预测，
  不是修改老师的游戏规则。未核验比赛服务器内部部署号或 Discord 私有公告。
- **办法 / Method：**

  | 定位 | 原预测的问题 | 修正后的行为 |
  |---|---|---|
  | `_bomb_schedule` | 预测炸弹会被另一颗炸弹提前连锁引爆 | 每颗炸弹按自身倒计时引爆 |
  | `_is_dangerous` | 现有火焰剩余危险时间额外加 1 步 | 使用 `explosion_map` 的剩余危险步数，不额外加 1 |
  | `_bomb_blocks` | 引爆当回合提前把炸弹格当作可通行 | 使用 `<= explosion_time`；先移动、后引爆，该回合仍阻路 |

- **保持不变 / Unchanged：** 模型权重、网络结构、目标输入特征、训练奖励均不变；
  手工目标 Q 加分保留 `1.00`，中心 Q 加分保留 `0.04`。没有重新训练。
- **结果 / Result：** 沿用上次已完成的 160 局配对对照：只修预测后规则组
  均分 4.40 → 4.35，混合对手组 2.35 → 3.25。混合组自杀仍为 3/20 局，
  规则组为 3/20 → 4/20，不能说已经解决自杀。45 项测试通过；这是规则
  一致性修复的采用，不是经过大规模验证的比赛最优声明。
- **队友后续操作 / Handoff：** `git pull origin main` 后，用
  `dueling_engine_aligned` 评测或观看。此目录仅用于推理，不能直接加
  `--train 1`。历史 `dueling_ddqn_agent`、`dueling_hunt_center_04` 和其他冻结
  版本未批量改写；如要在它们或你自己的新版本继续训练，应先移植以上三处
  修复并运行引擎对照测试，不要只改奖励或直接拿旧模型目录继续训练。
  箱子/对手仍按静态位置规划，动态堵路与安全回退仍是后续问题。
- **相关记录 / Record：** 修复和 160 局实验提交 `efead4d`；完整原始成绩见
  `experiments/records/engine_timing_ablation_20260916/`。本条为后续用户确认采用，
  不覆盖上一条当时“尚未采用”的实验记录。

### Engine timing correction and frozen-weight objective-bias ablation

- **Change:** Added the isolated `dueling_engine_aligned` inference candidate,
  copied from `dueling_hunt_center_04` with identical weights. Removed predicted
  chain explosions (absent from this engine), corrected existing-flame lifetime,
  and kept bomb tiles blocked through the detonation turn's movement phase.
  Historical agents, training rewards and the current submission remain unchanged.
- **Method:** A predeclared 2x2 evaluation crossed old/corrected prediction with
  objective Q bias 1.0/0.0. The learned objective feature, center bias 0.04 and
  repeat-position penalty stayed unchanged. Each arm played 20 rule3 games and
  20 mixed games against `feature_is_everything`, `deep_learning_killer` and
  `rule_based_agent` (160 games total), with paired initial-state hashes verified.
  No retraining. Added seven regression tests against actual engine methods;
  all 45 tests pass. All four evaluation processes finished successfully, and
  before/after checkpoint hashes match.
- **Result:** Mean scores (rule3 / mixed): baseline 4.40 / 2.35; corrected timing
  4.35 / 3.25; objective bias disabled 3.10 / 2.15; both changes 3.40 / 1.95.
  Corrected timing increased mixed tied-first rate from 20% to 40%, with the
  same 0.15 suicide events/game; rule3 suicide events rose from 0.15 to 0.20.
  Disabling the objective bias raised suicide events to 0.50 / 0.35 without
  the timing fix, or 0.45 / 0.35 with it. Retain the objective bias; the timing
  candidate merits a larger independent validation but is not auto-promoted.
  Mixed score difference +0.90 has a descriptive paired bootstrap 95% interval
  [0.15, 1.80], without multiple-comparison correction. This small suite does not
  establish general tournament superiority or solve suicide behavior.
- **Record:** `experiments/records/engine_timing_ablation_20260916/REPORT.md`,
  `summary.json`, locked `protocol.json`, and all eight raw match JSON files.
  Reproduce with `evaluation/engine_timing_ablation.py` and
  `evaluation/summarize_timing_ablation.py`. The baseline engine commit is
  `d514b60`; the exact source/weight hashes are recorded in the protocol.

## 2026-09-08

### Internal agents versus fixed external benchmarks

- **Change:** Added notebook 04 and a resumable tournament runner comparing
  `dueling_ddqn_agent`, `dueling_hunt_center_04`, `frozen_ddqn_day7`, and
  `hunt_center_04_mix_0800`, plus the external `deep_learning_killer` challenger,
  against the fixed external line-up
  `feature_is_everything + sarsa + q_learning`.
- **Method:** Ran classic matches at a fixed seat order on seeds 63001–63003,
  with 10 rounds per seed and no silenced agent errors. Stored each match as raw
  JSON, generated a combined report and CSV tables, and executed the notebook to
  embed tables, plots, and its data-derived conclusion. Added explicit
  four-player score tables and both aggregate and per-seed margins, defined as
  internal score minus the mean score of the three external agents.
- **Result:** Across 30 rounds per internal model, `dueling_hunt_center_04`
  ranked first internally at 3.30 score/round (seed SD 0.346), 0.367 kills,
  0.167 suicides, 2.267 invalid actions, and 2.36 ms/action. The added
  `hunt_center_04_mix_0800` scored 2.933 with a +0.167 external-average margin,
  but only a 0.20 win rate and 9.40 invalid actions per round; it does not replace
  the original model. The other scores remain 2.70 for `dueling_ddqn_agent` and
  2.10 for `frozen_ddqn_day7`. All twelve matches completed, and the executed
  notebook contains no error outputs. `external_deep_learning_killer` ranked
  fourth at 2.533 score with 0.90 survival and only 1.00 invalid action per round;
  its conservative safety did not displace the original hunt-center winner.
- **Commit:** this commit

### Isolated previous-cohort benchmark

- **Change:** Added evaluation-only, file-level symbolic-link adapters for the
  previous cohort's agents, using `external_*` names and a separate
  `results/external_benchmark/` result tree. Training callbacks are excluded;
  legacy pickle import aliases link only class-definition modules.
- **Method:** Added a reproducible gated runner for 3-round smoke tests,
  seed-61001 30-round screening, and five-seed confirmation. The runner first
  performs a rule-based framework preflight so missing shared dependencies are
  not misclassified as external-agent failures.
- **Result:** Six agents passed smoke. `baseline` is absent upstream; `dqn` and
  `dqn_5tile` cannot load their CUDA-serialized pickle models on the current CPU
  environment. The 30-round screen ranked `feature_is_everything` (5.767),
  `sarsa` (5.367), and `sarsa_lambda` (4.667) as the top three by mean score.
  The first confirmation run (`feature_is_everything`, seed 62001) is saved and
  the remaining confirmation matrix can be resumed. Summary tests pass 3/3.
- **Commit:** this commit

## 2026-08-26

### Day 18 — senior target guidance and active opening recovery

- **Change:** Studied the senior repository at commit `a7fe504` and adapted its
  strongest idea: use a BFS-style planner to mark an explicit target action for
  coins, productive crate positions, and opponents. Added a `1.00` inference
  bias and `0.50` training reward for following that target. A carefully scoped
  opening exception permits a productive one-branch bomb only before step 121
  when there are no bombs, explosions, or nearby opponents; normal play retains
  the two-branch safety requirement. Recent-position memory grew from 8 to 32.
- **Method:** Kept our Dueling Double DQN, five-step targets, PER, and D4 replay
  rather than copying the senior model or weights. Trained 50 target-curriculum
  rounds against peaceful opponents and 75 retention rounds against rule-based
  opponents. Compared both checkpoints on the same 50 unseen seed-`9801`
  rounds and replayed the original bad-opening seed `9602`.
- **Result:** On seed `9602`, the first bomb moved from step 135 to step 2 and
  bombs in the first 120 steps rose from 0 to 11; the promoted model survived
  all 400 steps. In the paired evaluation, the target stage averaged 3.94 score,
  0.28 kills, and 0.22 suicides versus the later stage's 2.84 score, 0.10 kills,
  and 0.08 suicides. The target stage was promoted because the later stage again
  became passive and raised invalid actions from 80 to 187. A second seed then
  exposed crate-adjacent targets where no safe bomb was possible; filtering
  every candidate target through a hypothetical escape simulation moved that
  seed's first bomb from step 129 to step 2 and first-120-step bombs from 0 to
  13. The final post-fix 50-round evaluation averaged 3.62 score, 0.20 kills,
  0.20 suicides, 37.82 bombs, and 29.84 destroyed crates per round at 4.36
  ms/action. All 27 tests pass.
  The promoted checkpoint SHA-256 is
  `709CCA4351F3BC5893F766B01B8C9335FA299D25331A275AE281B123DF31009A`.
  The rebuilt 282,920-byte submission ZIP passed CRC and syntax checks and has
  SHA-256
  `D6548DB49BDBAB6CB2668178073EA40CA7F0679E1675C31EAF5C981DF04D5E22`.
- **Commit:** this commit

### Day 17 — robust bomb safety and retraining

- **Change:** Replaced the one-path bomb-safety test with a hard requirement for
  at least two independent first-step escape branches. The planner now advances
  bombs triggered by chain reactions, treats current opponent tiles as blocked,
  and refuses to re-enable an unsafe bomb in its fallback mask.
- **Method:** Preserved the Day 16 model, added two focused safety tests, and
  continued it for 100 rounds against three rule-based agents on seed `9301`.
  Promotion used a paired 50-round evaluation on unseen seed `9401`, loading the
  exact pre-training and post-training checkpoints under identical code.
- **Result:** In the paired evaluation, suicides fell from 17/50 (`0.34` per
  round) to 13/50 (`0.26`, a 23.5% reduction), and invalid actions fell from 82
  to 46. Mean score changed from 2.42 to 2.24 and kills from 10 to 8; the 7.4%
  score tradeoff confirmed the safety training worked. A first visual run then
  exposed a 75-move narrow-area oscillation, so a `0.60` recent-position Q
  penalty was added only while the robust mask rejects `BOMB`. It cut the same
  replay regression to 5 moves. The final balanced 50-round controller averaged
  2.70 score, 0.30 kills, 0.34 suicides, and 2.47 ms/action, with 38 invalid
  actions. All 25 tests pass. The promoted checkpoint SHA-256 is
  `59355CD064EA4BE9996A7DA1A490CFC9772494DB6DFB1D5E894CDA550D87ACB8`.
  In the visible seed `9602` match, the promoted pink agent survived all 400
  steps, made 33 bombs with zero invalid actions, and its longest exact
  opposite-direction loop was three moves.
  The rebuilt 288,265-byte submission ZIP passed CRC and syntax checks and has
  SHA-256
  `80769C272D7F861903BCF2E76C5AF08B0313073BB8C4852580C19AB7C0F2E7E8`.
- **Commit:** this commit

### Day 16 — midgame crate-loop fix and visual regression

- **Change:** Added an explicit `safe_productive_bomb` action feature, rewarded
  safe bombs that can destroy at least one crate, and extended the recent-position
  revisit penalty to the midgame case where the agent is safely able to open a
  nearby crate. Promoted the rule-retention stage rather than the final training
  stage, and launched a visible match against three rule-based agents.
- **Method:** Migrated old 18- and 20-feature checkpoints to 21 features without
  changing existing input weights. Trained 50 peaceful safe-crate rounds, 100
  rule-retention rounds, and 50 mixed-retention rounds with disjoint seeds. The
  requested suicide multiplier remains `2.5`. Promotion compared every stage
  with the pre-fix checkpoint and replayed the originally observed seed `8901`.
- **Result:** On seed `8901`, the longest exact opposite-direction loop fell
  from 16 consecutive moves to 2 in the promoted rule-retention checkpoint. On
  the same 30-round seed `9002` comparison, it scored 132 versus the old model's
  109, collected 97 versus 94 coins, and made 7 versus 3 kills; suicides rose
  from 20 to 23, so safety remains the next optimization target. The later mixed
  stage was rejected after a 100-round check scored 348 versus the baseline's
  394. All 23 tests pass. The visible seed `9101` replay has no invalid action
  and its longest opposite-direction loop is 3 moves. The promoted checkpoint
  SHA-256 is
  `226ECF150BA869E3C9010067A04E840D7D7F42BA2C9A08CC2A32197125D84C67`.
  The rebuilt 286,671-byte submission ZIP passed CRC and syntax checks and has
  SHA-256
  `8397E845C0EB86C6EF09D1DA34BC4738A72605DF3300E0D0DA69E2988D4AE8C0`.
- **Commit:** this commit

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

## 2026-09-11

### Independent PPO implementation and Task 1–4 curriculum

- **Change:** Added `ppo_agent`, Actor-Critic/GAE/PPO tests, Task 1–4 frozen
  checkpoints, raw evaluations, and `06_ppo_tasks1_4.ipynb`.
- **Method:** Reused the current 22-feature DDQN encoding, safe mask and reward
  shaping, while using independent on-policy masked PPO. Trained 480 curriculum
  rounds and evaluated 360 held-out rounds, including the standard five-seed
  Task-4 protocol.
- **Result:** Task 1 reached 23.5 coins, Task 2 remained suicide-free, and Task
  3 scored 4.3 with 100% wins against passive agents. Task-4 scores were 2.333
  rule-based and 2.587 mixed, so this checkpoint is a functional baseline but
  not a mature competition policy.
- **Commit:** pending

## 2026-09-11

### Add frozen SARSA and linear-Q to notebook 03 comparison

- **Change:** Extended notebook 03 from four DDQN hunt-center variants to six
  frozen candidates by adding `frozen_sarsa_best` and
  `frozen_linear_q_final`.
- **Method:** Reused the original five seeds (32001–32005), both rule-based and
  mixed lineups, and 30 rounds per seed; ran 600 new inference-only rounds and
  re-executed the notebook. Kept the original DDQN selection rule scoped to the
  three hunt-center ablations.
- **Result:** Linear-Q/SARSA scored 3.067/3.020 against rule agents and
  3.900/3.740 in mixed games, below all DDQN variants. The original 0.08 DDQN
  recommendation remains unchanged.
- **Commit:** pending

## 2026-09-11

### Final controlled linear-Q stop-line experiment

- **Change:** Added `linear_q_final`, eight Task-4 interaction features, six
  alpha/n-step branches, the selected frozen checkpoint, and notebook 06.
- **Method:** Warm-started every branch from the same 61,973-update model,
  trained each for 200 rule-agent rounds, evaluated all on the same 50-round
  validation seed, then ran Task 1/2 regression and paired 100-round locked
  tests for the closest candidate and original baseline.
- **Result:** No candidate passed all gates. Alpha=0.003/n-step=8 led validation
  at 3.32 score, 0.22 suicides and 64% survival. On locked evaluation it scored
  2.98 versus baseline 3.02, while improving suicides 0.43→0.29 and survival
  34%→62%. The declared stop line was triggered.
- **Commit:** pending

## 2026-09-11

### Formal SARSA(lambda) sweep and low-alpha checkpoint selection

- **Change:** Added configurable SARSA hyperparameters, explicit bomb-escape
  shaping, lambda 0.4/0.7/0.9 training branches, two frozen selection stages,
  and `frozen_sarsa_best`; extended notebook 05 with formal results.
- **Method:** Continued all lambda branches from the same 73-episode model for
  40 paired-seed rule-agent rounds, evaluated them on 20 unseen rounds, then
  refined the selected lambda=0.4 branch for 40 rounds at alpha=0.001 and
  compared both checkpoints on another unseen 20-round seed.
- **Result:** Lambda=0.4 led the sweep at 3.8 mean score, 55% wins and 0.25
  suicides/round. Low-alpha refinement retained 3.8 mean score on the final
  selection seed while improving suicide rate from 0.50 to 0.35 and survival
  from 0.50 to 0.65; stage 2 was promoted.
- **Commit:** pending

## 2026-09-11

### SARSA(lambda) baseline and Tasks 1–4 debug curriculum

- **Change:** Added `agent_code/sarsa-lambda`, focused SARSA tests, a trained
  checkpoint, raw Task 1–4 results, and notebook 05.
- **Method:** Used linear accumulating-trace SARSA with the existing
  action-relative features/safety mask. Cached each nonterminal transition
  until the behaviour policy selected its actual next action. Trained a
  30/20/20-round coin/classic/opponent curriculum and evaluated on separate
  fixed seeds.
- **Result:** Eight focused tests and a three-round smoke run passed. Held-out
  Task 1 collected 50/50 coins; Task 2 averaged 34.4 crates without suicide;
  Task 3 won 10/10 against passive opponents; Task 4 exposed a 0.9 suicide rate
  and 0.2 win rate against rule-based agents. All learned weights stayed finite.
- **Commit:** pending

## 2026-09-15

### Controlled continuation notebook for hunt-center DDQN

- **Change:** Added `notebooks/08_continue_training_dddqn.ipynb` for a
  1000-round continuation from the frozen `dueling_hunt_center_04` checkpoint.
- **Method:** Prespecified an exact 40/30/20/10 opponent mixture in randomized
  20-round microbatches, isolated the writable training copy, froze checkpoints
  every 200 rounds, and separated paired 120-round development evaluation from
  a one-shot final holdout. Made preparation, training, and development cells
  idempotent through protocol, code, raw-stat, checkpoint-metadata, model-hash,
  and manifest validation before reusing existing artifacts. Converted every
  notebook Markdown cell to English for the final report version.
- **Result:** The notebook is reproducible and execution-gated; no training or
  evaluation is repeated when complete matching results exist, inconsistent or
  incomplete artifacts fail explicitly, and the original/configuration/0000
  checkpoint artifacts are never overwritten by the training or evaluation cells.
  The one-shot 120-round final holdout for the preselected 0800 checkpoint scored
  3.583 with a 0.308 win rate, 0.150 suicides, and 5.150 invalid actions per round;
  its tournament metrics did not replicate the development gain, so the notebook
  records the decision to retain the original `dueling_hunt_center_04`.
- **Commit:** pending

### Formal mixed-opponent benchmark notebook

- **Change:** Added notebook 09 and a dedicated resumable runner for five
  candidate agents against `external_feature_is_everything`,
  `external_deep_learning_killer`, and `rule_based_agent`.
- **Method:** Locked five previously unused seeds (64001–64005), 30 rounds per
  seed, fixed candidate/opponent slots, and 150 rounds per candidate. Each raw
  result receives a full-protocol provenance sidecar and SHA-256; analysis uses
  a paired seed-cluster bootstrap rather than treating all rounds as independent.
- **Result:** All 750 rounds completed without agent failures. `dueling_ddqn_agent`
  had the highest score point estimate at 2.907 versus 2.820 for
  `dueling_hunt_center_04`, but the paired seed-cluster difference was only
  +0.087 with 95% CI [-0.513, 0.660]. The 0800 continuation again showed high
  invalid actions (8.407/round), while both linear candidates were clearly
  weaker. The report retains `dueling_hunt_center_04` for submission because the
  small DDQN point-estimate lead does not overturn the broader locked evidence.
- **Commit:** pending

## Entry template

```markdown
## YYYY-MM-DD

### Short update title

- **Change:**
- **Method:**
- **Result:**
- **Commit:**
```
