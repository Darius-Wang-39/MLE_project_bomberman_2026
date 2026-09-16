"""Locked validation selection, independent holdout comparison and raw archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

TAGS = ("baseline", "p2", "p3", "both", "trained100", "trained200")
LINEUPS = ("rule3", "mixed")


def load_results(arena, tag, expected_rounds):
    summary = json.loads((arena / "results" / f"{tag}_summary.json").read_text(encoding="utf-8"))
    if len(summary) != 2 or any(r["rounds"] != expected_rounds for r in summary):
        raise ValueError(f"Incomplete group {tag}")
    raw = {}
    for lineup in LINEUPS:
        raw[lineup] = json.loads((arena / "results" / f"{tag}_{lineup}.json").read_text(encoding="utf-8"))
    return summary, raw


def selection(arena):
    rows, reference = [], None
    for tag in TAGS:
        summary, raw = load_results(arena, tag, 10)
        initial = {k: v["evaluation"]["initial_conditions"] for k, v in raw.items()}
        if reference is None:
            reference = initial
        if initial != reference:
            raise ValueError(f"Unpaired validation initial conditions: {tag}")
        row = {"tag": tag, "mean_score": sum(r["mean_score"] for r in summary) / 2, "lineups": summary}
        rows.append(row)
    # Selection rule is fixed before any results, with ties favoring earlier tags.
    chosen = max(rows, key=lambda r: r["mean_score"])
    challenger = max(rows[1:], key=lambda r: r["mean_score"])
    return {"validation": rows, "selected_including_baseline": chosen["tag"],
            "best_modified_challenger": challenger["tag"],
            "validation_initial_conditions_matched": True}


def holdout(arena):
    baseline, base_raw = load_results(arena, "holdout_baseline", 20)
    challenger, new_raw = load_results(arena, "holdout_challenger", 20)
    differences = {}
    for lineup in LINEUPS:
        if base_raw[lineup]["evaluation"]["initial_conditions"] != new_raw[lineup]["evaluation"]["initial_conditions"]:
            raise ValueError("Unpaired holdout initial conditions")
        a = [r["agents"]["candidate"]["score"] for r in base_raw[lineup]["by_round"].values()]
        b = [r["agents"]["candidate"]["score"] for r in new_raw[lineup]["by_round"].values()]
        delta = np.asarray(b) - np.asarray(a)
        rng = np.random.default_rng(919999)
        boot = rng.choice(delta, (10000, len(delta)), replace=True).mean(axis=1)
        differences[lineup] = {"paired_mean_score_difference": float(delta.mean()),
                               "descriptive_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist()}
    return {"baseline": baseline, "challenger": challenger, "differences": differences,
            "initial_conditions_matched": True}


def archive(arena, output):
    report = selection(arena)
    report["holdout"] = holdout(arena)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(arena / "protocol.json", output / "protocol.json")
    for path in (arena / "results").glob("*.json"):
        shutil.copy2(path, output / path.name)
    import torch
    (output / "checkpoints").mkdir()
    original = torch.load(arena / "agent_code/dueling_priority23/dueling_ddqn_model.pt",
                          map_location="cpu", weights_only=True)
    metadata = []
    for count in (100, 200):
        path = arena / "checkpoints" / f"trained_{count:04d}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        shutil.copy2(path, output / "checkpoints" / path.name)
        metadata.append({"rounds": count, "flags": [True, True],
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "environment_steps_counter": checkpoint["environment_steps"],
                         "added_counter_steps": checkpoint["environment_steps"] - original["environment_steps"],
                         "learner_updates": checkpoint["learner_updates"],
                         "added_updates": checkpoint["learner_updates"] - original["learner_updates"],
                         "input_width": checkpoint["policy_state"]["encoder.0.weight"].shape[1]})
    training = [json.loads(p.read_text()) for p in (arena / "results").glob("train_block_*.json")]
    report["training"] = {
        "rounds": sum(len(p["by_round"]) for p in training),
        "engine_recorded_actions": sum(p["by_agent"]["candidate"]["steps"] for p in training),
        "checkpoints": metadata,
        "counter_note": "Trainer counter includes terminal callbacks; it is not a count of distinct game actions.",
    }
    (output / "REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# 2026-09-17 — P2/P3 修改、200 局续训和独立验证", "",
             "修改和训练于 9 月 16 日开始，结果于 9 月 17 日完成归档。",
             f"训练共 {report['training']['rounds']} 局，引擎记录 {report['training']['engine_recorded_actions']} 次动作。",
             "100/200 局权重及其校验值随本目录保存；训练计数含终止回调，不等同于不同的游戏动作。", "",
             "## 验证集（用于选版本）", "",
             "每版本 10 局规则对手 + 10 局学长混合对手。共享初始地图和出生位置，均不更新权重。", "",
             "| 版本 | 规则组均分 | 混合组均分 | 两组平均 |", "|---|---:|---:|---:|"]
    for row in report["validation"]:
        lines.append(f"| {row['tag']} | {row['lineups'][0]['mean_score']:.2f} | "
                     f"{row['lineups'][1]['mean_score']:.2f} | {row['mean_score']:.2f} |")
    lines += ["", f"按预定均分规则选出的版本：`{report['selected_including_baseline']}`。",
              f"最佳修改版（独立测试挑战者）：`{report['best_modified_challenger']}`。", "",
              "baseline=此前采用的引擎修正版（26 列零扩展对照）；p2=只改资源效率；p3=只改堵路风险；",
              "both=两项都改但尚未训练；trained100/200=两项都改后追加 100/200 局训练。", "",
              "## 独立测试集（不用于选版本）", "",
              "每版本每阵容 20 局，种子与训练、验证分开；测试集只比较基线和预先选出的最佳修改版。", "",
              "| 阵容 / 版本 | 均分 | 第一名率（含并列） | 自杀事件/局 | 金币/局 | 炸弹/局 | 无效动作/局 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label in ("baseline", "challenger"):
        for r in report["holdout"][label]:
            lines.append(f"| {r['lineup']} / {label} | {r['mean_score']:.2f} | {r['win_rate']:.0%} | "
                         f"{r['suicides_per_round']:.2f} | {r['coins_per_round']:.2f} | "
                         f"{r['bombs_per_round']:.2f} | {r['invalid_per_round']:.2f} |")
    lines += ["", "## 边界", "",
              "第 2、3 项是实验性改进，不是像第 1 项一样已确定的引擎错误修复。",
              "未增加目标 Q 手工偏置，保留 1.0；中心偏置保留 0.04，自杀系数保留 2.5。",
              "新增特征零初始化输入权重、保留旧网络权重；新优化器和经验池在训练开始时重置，分段间持续保留。",
              "奖励常数未改，但与目标距离/炸弹收益有关的奖励使用新的特征计算。",
              "对手到达时间只用于有限的狭窄出口压力测试，不代表知道对手实际动作。不能保证不会被堵或自杀。",
              "每组样本较小，且只有固定的两种阵容；不能据此保证正式比赛排名。",
              "完整配置、逐局成绩和初始状态哈希随报告保存；当次实验不自动替换正式候选。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        archive(args.arena.resolve(), args.output.resolve())
    else:
        print(json.dumps(selection(args.arena.resolve()), indent=2))
