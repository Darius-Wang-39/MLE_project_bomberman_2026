"""Verify pairing and archive all raw data for the timing/bias experiment."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

try:
    from .engine_timing_ablation import ARMS, LINEUPS
    from .summarize_stats import summarize_match
except ImportError:
    from engine_timing_ablation import ARMS, LINEUPS
    from summarize_stats import summarize_match


def paired_interval(differences):
    """Descriptive paired bootstrap; not a multiple-comparison significance test."""
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(916999)
    boot = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return [float(v) for v in np.quantile(boot, [0.025, 0.975])]


def collect(arena):
    report = {"training": False, "initial_conditions_matched": True, "lineups": {}}
    for lineup in LINEUPS:
        reference_initial, reference_scores = None, None
        rows = []
        for arm in ARMS:
            path = arena / "results" / f"{arm}_{lineup}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            initial = payload["evaluation"]["initial_conditions"]
            scores = [row["agents"]["candidate"]["score"] for row in payload["by_round"].values()]
            if len(scores) != 20 or len(initial) != 20:
                raise ValueError("Only complete predeclared 20-round groups may enter this report")
            if reference_initial is None:
                reference_initial, reference_scores = initial, scores
            if initial != reference_initial:
                raise ValueError(f"Initial boards/positions do not match for {arm}/{lineup}")
            differences = [score - base for score, base in zip(scores, reference_scores)]
            metrics = next(row for row in summarize_match([path]) if row["agent"] == "candidate")
            rows.append({"arm": arm, **metrics,
                         "score_difference_from_baseline": float(np.mean(differences)),
                         "paired_bootstrap_95_interval": paired_interval(differences)})
        report["lineups"][lineup] = rows
    return report


def archive(arena, output):
    report = collect(arena)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("Refusing to overwrite an archived experiment")
    shutil.copy2(arena / "protocol.json", output / "protocol.json")
    for arm in ARMS:
        for lineup in LINEUPS:
            shutil.copy2(arena / "results" / f"{arm}_{lineup}.json", output)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# 2026-09-16 — 爆炸预测 × 手工目标加分对照", "",
             "基线：`dueling_hunt_center_04`。四组共 160 局，不训练、不改权重。",
             "每种对手阵容 20 个固定种子，四组初始地图、隐藏金币和出生位置的哈希全部核对一致。",
             "本次只覆盖固定对手组合和座位顺序，不能直接推断最终比赛排名。", "",
             "- baseline：原预测、目标 Q 加分 1.0。",
             "- timing：修正预测、目标 Q 加分 1.0。",
             "- no_bias：原预测、目标 Q 加分 0.0。",
             "- both：修正预测、目标 Q 加分 0.0。", "",
             "修正仅含：取消虚构连锁引爆、纠正现有火焰剩余步数、炸弹格在引爆回合行动阶段仍阻路。",
             "保留中心偏置 0.04、重复走位惩罚、目标输入特征和全部已学权重。", ""]
    for lineup, rows in report["lineups"].items():
        lines += [f"## {lineup}", "",
                  "| 组别 | 平均分 | 第一名率（含并列） | 自杀事件/局 | 无效动作/局 | 得分差及配对 95% 区间 |",
                  "|---|---:|---:|---:|---:|---|"]
        for row in rows:
            low, high = row["paired_bootstrap_95_interval"]
            lines.append(f"| {row['arm']} | {row['mean_score']:.2f} | {row['win_rate']:.0%} | "
                         f"{row['suicides_per_round']:.2f} | {row['invalid_per_round']:.2f} | "
                         f"{row['score_difference_from_baseline']:+.2f} [{low:+.2f}, {high:+.2f}] |")
        lines.append("")
    lines += ["## 解释边界", "",
              "rule3 = 三个 rule_based_agent；mixed = feature_is_everything + deep_learning_killer + rule_based_agent。",
              "得分差均对照同阵容 baseline。区间为 10,000 次配对 bootstrap 的描述性区间，未做多重比较校正。",
              "每组每阵容只有 20 局；区间跨过零时，不能认定总体得分确实提高。",
              "自杀列是引擎记录的 KILLED_SELF 事件/局；第一名率按最终得分且包含并列，不等于存活率。",
              "对手保留原策略，只为学长 pickle 权重加 CPU 加载适配；每局重置环境和随机决策种子。",
              "预测器仍将箱子、对手位置视为静态；这次没有修正安全回退、动态堵路或重新训练。", "",
              "## 复现", "",
              "从仓库根目录运行（新建的 arena 路径不可已存在）：", "",
              "```text",
              "python evaluation/engine_timing_ablation.py --prepare --arena <arena> --senior-root <senior-checkout>",
              "python evaluation/engine_timing_ablation.py --arena <arena> --arm baseline",
              "python evaluation/engine_timing_ablation.py --arena <arena> --arm timing",
              "python evaluation/engine_timing_ablation.py --arena <arena> --arm no_bias",
              "python evaluation/engine_timing_ablation.py --arena <arena> --arm both",
              "python evaluation/summarize_timing_ablation.py --arena <arena> --output <archive>",
              "```", "",
              "本目录保存完整原始比赛 JSON、初始条件哈希、源码/权重哈希和汇总，便于队友复查。",
              "每组最后一个预设种子的录像保存在隔离 arena/replays；未按输赢挑选。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive(args.arena.resolve(), args.output.resolve())
