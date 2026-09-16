# 2026-09-16 — 爆炸预测 × 手工目标加分对照

基线：`dueling_hunt_center_04`。四组共 160 局，不训练、不改权重。
每种对手阵容 20 个固定种子，四组初始地图、隐藏金币和出生位置的哈希全部核对一致。
本次只覆盖固定对手组合和座位顺序，不能直接推断最终比赛排名。

- baseline：原预测、目标 Q 加分 1.0。
- timing：修正预测、目标 Q 加分 1.0。
- no_bias：原预测、目标 Q 加分 0.0。
- both：修正预测、目标 Q 加分 0.0。

修正仅含：取消虚构连锁引爆、纠正现有火焰剩余步数、炸弹格在引爆回合行动阶段仍阻路。
保留中心偏置 0.04、重复走位惩罚、目标输入特征和全部已学权重。

## 本次结论

继续验证“只修预测、保留目标加分 1.0”的候选。它在混合组均分提高 0.90，
第一名率从 20% 到 40%，规则组得分基本持平。关闭加分的两个版本均不推荐直接使用。
目前未替换 `dueling_hunt_center_04` 或已有提交包，也没有重新训练。

安全问题未解决：只修预测在混合组仍有 3/20 局自杀，规则组从原版 3/20 到 4/20。
更准确的预测改变了动作筛选和网络输入，但不意味着旧权重已经适应，或动态堵路问题消失。
建议先在新的地图种子、轮换座位的更大验证集确认，再决定是否正式采用。

全部 45 项单元测试通过，其中 7 项新增测试直接调用引擎核对时序。
四组评测进程均成功结束，模型权重前后 SHA-256 一致，没有训练。

## rule3

| 组别 | 平均分 | 第一名率（含并列） | 自杀事件/局 | 无效动作/局 | 得分差及配对 95% 区间 |
|---|---:|---:|---:|---:|---|
| baseline | 4.40 | 50% | 0.15 | 1.75 | +0.00 [+0.00, +0.00] |
| timing | 4.35 | 45% | 0.20 | 1.70 | -0.05 [-0.90, +0.95] |
| no_bias | 3.10 | 40% | 0.50 | 1.80 | -1.30 [-2.65, +0.00] |
| both | 3.40 | 45% | 0.45 | 1.50 | -1.00 [-2.30, +0.30] |

## mixed

| 组别 | 平均分 | 第一名率（含并列） | 自杀事件/局 | 无效动作/局 | 得分差及配对 95% 区间 |
|---|---:|---:|---:|---:|---|
| baseline | 2.35 | 20% | 0.15 | 3.50 | +0.00 [+0.00, +0.00] |
| timing | 3.25 | 40% | 0.15 | 3.00 | +0.90 [+0.15, +1.80] |
| no_bias | 2.15 | 15% | 0.35 | 0.90 | -0.20 [-1.45, +1.10] |
| both | 1.95 | 15% | 0.35 | 1.90 | -0.40 [-1.60, +0.75] |

## 解释边界

rule3 = 三个 rule_based_agent；mixed = feature_is_everything + deep_learning_killer + rule_based_agent。
得分差均对照同阵容 baseline。区间为 10,000 次配对 bootstrap 的描述性区间，未做多重比较校正。
每组每阵容只有 20 局；区间跨过零时，不能认定总体得分确实提高。
自杀列是引擎记录的 KILLED_SELF 事件/局；第一名率按最终得分且包含并列，不等于存活率。
对手保留原策略，只为学长 pickle 权重加 CPU 加载适配；每局重置环境和随机决策种子。
预测器仍将箱子、对手位置视为静态；这次没有修正安全回退、动态堵路或重新训练。

## 复现

从仓库根目录运行（新建的 arena 路径不可已存在）：

```text
python evaluation/engine_timing_ablation.py --prepare --arena <arena> --senior-root <senior-checkout>
python evaluation/engine_timing_ablation.py --arena <arena> --arm baseline
python evaluation/engine_timing_ablation.py --arena <arena> --arm timing
python evaluation/engine_timing_ablation.py --arena <arena> --arm no_bias
python evaluation/engine_timing_ablation.py --arena <arena> --arm both
python evaluation/summarize_timing_ablation.py --arena <arena> --output <archive>
```

本目录保存完整原始比赛 JSON、初始条件哈希、源码/权重哈希和汇总，便于队友复查。
每组最后一个预设种子的录像保存在隔离 arena/replays；未按输赢挑选。
