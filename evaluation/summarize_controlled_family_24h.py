"""Aggregate, visualize, and report the controlled five-family experiment."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.summarize_stats import summarize_match


RESULT_ROOT = ROOT / "results" / "controlled_family_24h"
RAW = RESULT_ROOT / "raw"
CHECKPOINTS = RESULT_ROOT / "checkpoints"
SEEDS = (71001, 72001, 73001)
AGENTS = (
    "controlled_linear_q",
    "controlled_sarsa_lambda",
    "controlled_ppo",
    "controlled_dddqn_nod4",
    "controlled_dddqn_d4",
)
LABELS = {
    "controlled_linear_q": "Linear Q",
    "controlled_sarsa_lambda": "SARSA(lambda)",
    "controlled_ppo": "PPO (no D4)",
    "controlled_dddqn_nod4": "DDDQN (no D4)",
    "controlled_dddqn_d4": "DDDQN (+D4)",
}
COLORS = {
    "controlled_linear_q": "#4C78A8",
    "controlled_sarsa_lambda": "#F58518",
    "controlled_ppo": "#54A24B",
    "controlled_dddqn_nod4": "#E45756",
    "controlled_dddqn_d4": "#B279A2",
}
CURVE = (("anchor", 0), ("task4_025", 25), ("task4_050", 50), ("task4_100", 100))


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def stats_path(seed: int, agent: str, checkpoint: str, suite: str) -> Path:
    return RAW / "evaluation" / f"seed_{seed}" / agent / checkpoint / f"{suite}.json"


def row_for(path: Path, agent: str) -> dict:
    rows = {row["agent"]: row for row in summarize_match([path])}
    if agent not in rows:
        raise KeyError(f"Missing {agent} in {path}")
    row = dict(rows[agent])
    return {key: (None if isinstance(value, float) and math.isnan(value) else value)
            for key, value in row.items()}


def aggregate_for(paths: list[Path], agent: str) -> dict:
    rows = {row["agent"]: row for row in summarize_match(paths)}
    row = dict(rows[agent])
    return {key: (None if isinstance(value, float) and math.isnan(value) else value)
            for key, value in row.items()}


def summarize_curve() -> tuple[dict, list[dict]]:
    summary = {}
    csv_rows = []
    for agent in AGENTS:
        summary[agent] = {}
        for checkpoint, budget in CURVE:
            per_seed = []
            paths = []
            for seed in SEEDS:
                path = stats_path(seed, agent, checkpoint, "task4_rule")
                paths.append(path)
                row = row_for(path, agent)
                per_seed.append({"training_seed": seed, **row})
                csv_rows.append({
                    "agent": agent,
                    "training_seed": seed,
                    "checkpoint": checkpoint,
                    "task4_rounds": budget,
                    **{key: value for key, value in row.items() if key != "agent"},
                })
            aggregate = aggregate_for(paths, agent)
            scores = [row["mean_score"] for row in per_seed]
            summary[agent][checkpoint] = {
                "task4_rounds": budget,
                "aggregate": aggregate,
                "per_training_seed": per_seed,
                "training_seed_score_mean": statistics.mean(scores),
                "training_seed_score_sd": statistics.stdev(scores),
                "training_seed_score_range": [min(scores), max(scores)],
            }
        anchor = summary[agent]["anchor"]["training_seed_score_mean"]
        for checkpoint, _ in CURVE:
            summary[agent][checkpoint]["score_gain_from_anchor"] = (
                summary[agent][checkpoint]["training_seed_score_mean"] - anchor
            )
    return summary, csv_rows


def summarize_capabilities() -> dict:
    output = {}
    for agent in AGENTS:
        output[agent] = {}
        for checkpoint in ("anchor", "task4_100"):
            output[agent][checkpoint] = {}
            for suite in ("task1_coin", "task2_solo", "task3_peaceful", "task4_rule"):
                paths = [stats_path(seed, agent, checkpoint, suite) for seed in SEEDS]
                output[agent][checkpoint][suite] = aggregate_for(paths, agent)
        paths = [stats_path(seed, agent, "task4_100", "task4_mixed") for seed in SEEDS]
        output[agent]["task4_100"]["task4_mixed"] = aggregate_for(paths, agent)
        output[agent]["delta"] = {
            "task1_coins": (
                output[agent]["task4_100"]["task1_coin"]["coins_per_round"]
                - output[agent]["anchor"]["task1_coin"]["coins_per_round"]
            ),
            "task2_crates": (
                output[agent]["task4_100"]["task2_solo"]["crates_per_round"]
                - output[agent]["anchor"]["task2_solo"]["crates_per_round"]
            ),
            "task3_kills": (
                output[agent]["task4_100"]["task3_peaceful"]["kills_per_round"]
                - output[agent]["anchor"]["task3_peaceful"]["kills_per_round"]
            ),
            "task4_score": (
                output[agent]["task4_100"]["task4_rule"]["mean_score"]
                - output[agent]["anchor"]["task4_rule"]["mean_score"]
            ),
            "task4_suicides": (
                output[agent]["task4_100"]["task4_rule"]["suicides_per_round"]
                - output[agent]["anchor"]["task4_rule"]["suicides_per_round"]
            ),
        }
    return output


def training_record(seed: int, agent: str, stage: str) -> dict:
    path = RAW / "training" / f"seed_{seed}" / agent / f"{stage}.run.json"
    return json.loads(path.read_text())


def training_steps(seed: int, agent: str, stage: str) -> int:
    path = RAW / "training" / f"seed_{seed}" / agent / f"{stage}.json"
    payload = json.loads(path.read_text())
    return int(payload["by_agent"][agent]["steps"])


def summarize_costs() -> list[dict]:
    rows = []
    base_stages = ("a1_coin", "a2_solo", "a3_peaceful")
    all_stages = (*base_stages, "task4_025", "task4_050", "task4_100")
    for agent in AGENTS:
        per_seed = []
        for seed in SEEDS:
            base_seconds = sum(training_record(seed, agent, stage)["elapsed_seconds"] for stage in base_stages)
            b100_seconds = training_record(seed, agent, "task4_100")["elapsed_seconds"]
            base_steps = sum(training_steps(seed, agent, stage) for stage in base_stages)
            b100_steps = training_steps(seed, agent, "task4_100")
            all_seconds = sum(training_record(seed, agent, stage)["elapsed_seconds"] for stage in all_stages)
            per_seed.append((base_seconds, b100_seconds, base_steps, b100_steps, all_seconds))
        checkpoint_path = CHECKPOINTS / f"seed_{SEEDS[0]}" / agent / "task4_100"
        model_files = list(checkpoint_path.iterdir())
        if len(model_files) != 1:
            raise ValueError(f"Expected one checkpoint in {checkpoint_path}")
        b100_eval = aggregate_for(
            [stats_path(seed, agent, "task4_100", "task4_rule") for seed in SEEDS], agent
        )
        rows.append({
            "agent": agent,
            "anchor_training_seconds_mean": statistics.mean(row[0] for row in per_seed),
            "task4_100_increment_seconds_mean": statistics.mean(row[1] for row in per_seed),
            "selected_480_round_path_seconds_mean": statistics.mean(row[0] + row[1] for row in per_seed),
            "all_555_round_branches_seconds_mean": statistics.mean(row[4] for row in per_seed),
            "selected_path_environment_steps_mean": statistics.mean(row[2] + row[3] for row in per_seed),
            "task4_100_environment_steps_mean": statistics.mean(row[3] for row in per_seed),
            "checkpoint_bytes": model_files[0].stat().st_size,
            "task4_100_mean_action_ms": b100_eval["mean_action_ms"],
        })
    return rows


def paired_contrasts(curve: dict) -> dict:
    pairs = {
        "dddqn_nod4_minus_ppo": ("controlled_dddqn_nod4", "controlled_ppo"),
        "dddqn_nod4_minus_linear": ("controlled_dddqn_nod4", "controlled_linear_q"),
        "dddqn_nod4_minus_sarsa": ("controlled_dddqn_nod4", "controlled_sarsa_lambda"),
        "dddqn_d4_minus_nod4": ("controlled_dddqn_d4", "controlled_dddqn_nod4"),
    }
    output = {}
    for name, (left, right) in pairs.items():
        output[name] = {}
        for checkpoint, budget in CURVE:
            left_scores = [row["mean_score"] for row in curve[left][checkpoint]["per_training_seed"]]
            right_scores = [row["mean_score"] for row in curve[right][checkpoint]["per_training_seed"]]
            differences = [a - b for a, b in zip(left_scores, right_scores)]
            output[name][checkpoint] = {
                "task4_rounds": budget,
                "paired_training_seed_differences": differences,
                "mean_difference": statistics.mean(differences),
                "positive_seed_count": sum(value > 0 for value in differences),
            }
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(curve: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    budgets = [budget for _, budget in CURVE]
    for agent in AGENTS:
        seed_scores = np.asarray([
            [row["mean_score"] for row in curve[agent][checkpoint]["per_training_seed"]]
            for checkpoint, _ in CURVE
        ]).T
        mean = seed_scores.mean(axis=0)
        for values in seed_scores:
            axes[0].plot(budgets, values, color=COLORS[agent], alpha=0.20, linewidth=1)
        axes[0].plot(budgets, mean, marker="o", color=COLORS[agent], label=LABELS[agent])
        axes[1].plot(budgets, mean - mean[0], marker="o", color=COLORS[agent], label=LABELS[agent])
    axes[0].set_title("Held-out Task-4 score")
    axes[1].set_title("Score change from anchor")
    for axis in axes:
        axis.set_xlabel("Additional rule-agent training rounds")
        axis.grid(alpha=0.25)
        axis.set_xticks(budgets)
    axes[0].set_ylabel("Mean score per round")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Delta score")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(RESULT_ROOT / f"figure_family_learning_curve.{suffix}", dpi=200)
    plt.close(fig)


def plot_capability_delta(capabilities: dict) -> None:
    metrics = ("task1_coins", "task2_crates", "task3_kills", "task4_score", "task4_suicides")
    metric_labels = ("T1 coins", "T2 crates", "T3 kills", "T4 score", "T4 suicides")
    matrix = np.asarray([[capabilities[agent]["delta"][metric] for agent in AGENTS] for metric in metrics])
    scales = np.max(np.abs(matrix), axis=1)
    normalized = matrix / np.where(scales == 0, 1, scales)[:, None]
    fig, axis = plt.subplots(figsize=(9.5, 4.2))
    image = axis.imshow(normalized, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(AGENTS)), [LABELS[agent] for agent in AGENTS], rotation=25, ha="right")
    axis.set_yticks(range(len(metrics)), metric_labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:+.2f}", ha="center", va="center", fontsize=9)
    axis.set_title("Capability change after 100 Task-4 training rounds (B100 - anchor)")
    fig.colorbar(image, ax=axis, label="Row-normalized signed change")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(RESULT_ROOT / f"figure_capability_delta.{suffix}", dpi=200)
    plt.close(fig)


def plot_capability_profile(capabilities: dict) -> None:
    metrics = (
        ("task1_coin", "coins_per_round", "T1 coins"),
        ("task2_solo", "crates_per_round", "T2 crates"),
        ("task3_peaceful", "kills_per_round", "T3 kills"),
        ("task4_rule", "mean_score", "T4 score"),
    )
    raw = {}
    for checkpoint in ("anchor", "task4_100"):
        raw[checkpoint] = np.asarray([
            [capabilities[agent][checkpoint][suite][field] for agent in AGENTS]
            for suite, field, _ in metrics
        ])
    row_max = np.maximum(raw["anchor"].max(axis=1), raw["task4_100"].max(axis=1))
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8), sharey=True, constrained_layout=True)
    for axis, checkpoint, title in zip(
        axes, ("anchor", "task4_100"), ("Anchor after 380 rounds", "After +100 Task-4 rounds")
    ):
        matrix = raw[checkpoint]
        normalized = matrix / np.where(row_max == 0, 1, row_max)[:, None]
        image = axis.imshow(normalized, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        axis.set_xticks(range(len(AGENTS)), [LABELS[agent] for agent in AGENTS], rotation=28, ha="right")
        axis.set_yticks(range(len(metrics)), [label for _, _, label in metrics])
        axis.set_title(title)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=8.5)
    fig.colorbar(image, ax=axes, label="Row-normalized capability", shrink=0.86, pad=0.02)
    fig.suptitle("Four-task capability decomposition (absolute held-out metrics)")
    for suffix in ("png", "pdf"):
        fig.savefig(RESULT_ROOT / f"figure_capability_profile.{suffix}", dpi=200)
    plt.close(fig)


def plot_d4_effect(curve: dict) -> None:
    budgets = [budget for _, budget in CURVE]
    d4 = np.asarray([
        [row["mean_score"] for row in curve["controlled_dddqn_d4"][checkpoint]["per_training_seed"]]
        for checkpoint, _ in CURVE
    ]).T
    nod4 = np.asarray([
        [row["mean_score"] for row in curve["controlled_dddqn_nod4"][checkpoint]["per_training_seed"]]
        for checkpoint, _ in CURVE
    ]).T
    differences = d4 - nod4
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    for seed, values in zip(SEEDS, differences):
        axis.plot(budgets, values, marker="o", alpha=0.5, label=f"train seed {seed}")
    axis.plot(budgets, differences.mean(axis=0), color="black", marker="o", linewidth=2.2, label="mean")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(budgets)
    axis.set_xlabel("Additional rule-agent training rounds")
    axis.set_ylabel("Score difference: DDDQN+D4 - DDDQN-noD4")
    axis.set_title("Paired D4 augmentation effect")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(RESULT_ROOT / f"figure_d4_effect.{suffix}", dpi=200)
    plt.close(fig)


def markdown_report(curve: dict, capabilities: dict, costs: list[dict], contrasts: dict) -> str:
    lines = [
        "# Controlled five-family experiment",
        "",
        "Three independent training seeds were run for five agents. Each agent received a 380-round "
        "foundation curriculum, followed by independent 25/50/100-round Task-4 branches. All agents "
        "shared the 17-feature representation, safety mask, event reward, potential shaping, training "
        "lineups and evaluation seeds. The two DDDQN variants differed only in replay-time D4 augmentation.",
        "",
        "## Task-4 learning curve",
        "",
        "| agent | anchor | B25 | B50 | B100 | B100-anchor | B100 suicides | B100 survival |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for agent in AGENTS:
        values = [curve[agent][checkpoint]["aggregate"]["mean_score"] for checkpoint, _ in CURVE]
        b100 = curve[agent]["task4_100"]["aggregate"]
        lines.append(
            f"| {LABELS[agent]} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | "
            f"{values[3]:.3f} | {values[3]-values[0]:+.3f} | "
            f"{b100['suicides_per_round']:.3f} | {b100['survival_rate']:.3f} |"
        )
    lines += ["", "## B100 generalization and cost", "",
              "| agent | mixed score | selected-path time (s) | env steps | checkpoint KiB | action ms |",
              "|---|---:|---:|---:|---:|---:|"]
    cost_by_agent = {row["agent"]: row for row in costs}
    for agent in AGENTS:
        cost = cost_by_agent[agent]
        mixed = capabilities[agent]["task4_100"]["task4_mixed"]["mean_score"]
        lines.append(
            f"| {LABELS[agent]} | {mixed:.3f} | {cost['selected_480_round_path_seconds_mean']:.1f} | "
            f"{cost['selected_path_environment_steps_mean']:.0f} | {cost['checkpoint_bytes']/1024:.1f} | "
            f"{cost['task4_100_mean_action_ms']:.3f} |"
        )
    lines += [
        "",
        "## Four-task capability decomposition",
        "",
        "| agent | T1 coins A/B100 | T2 crates A/B100 | T3 kills A/B100 | T4 score A/B100 |",
        "|---|---:|---:|---:|---:|",
    ]
    for agent in AGENTS:
        anchor = capabilities[agent]["anchor"]
        b100 = capabilities[agent]["task4_100"]
        lines.append(
            f"| {LABELS[agent]} | {anchor['task1_coin']['coins_per_round']:.2f} / "
            f"{b100['task1_coin']['coins_per_round']:.2f} | "
            f"{anchor['task2_solo']['crates_per_round']:.2f} / {b100['task2_solo']['crates_per_round']:.2f} | "
            f"{anchor['task3_peaceful']['kills_per_round']:.2f} / {b100['task3_peaceful']['kills_per_round']:.2f} | "
            f"{anchor['task4_rule']['mean_score']:.2f} / {b100['task4_rule']['mean_score']:.2f} |"
        )
    d4 = contrasts["dddqn_d4_minus_nod4"]["task4_100"]
    nod4_ppo = contrasts["dddqn_nod4_minus_ppo"]["task4_100"]
    lines += [
        "",
        "## Prespecified contrasts",
        "",
        f"- At B100, DDDQN without D4 minus PPO: {nod4_ppo['mean_difference']:+.3f} score/round; "
        f"positive in {nod4_ppo['positive_seed_count']}/3 training seeds.",
        f"- At B100, DDDQN with D4 minus DDDQN without D4: {d4['mean_difference']:+.3f} "
        f"score/round; positive in {d4['positive_seed_count']}/3 training seeds.",
        "- With only three independent training seeds, these are descriptive paired effects, not "
        "high-powered significance tests.",
        "",
        "## Interpretation boundary",
        "",
        "The experiment compares development families under equal external exposure, not bare optimizers. "
        "Family-defining mechanisms (eligibility traces, PPO/GAE, and DDDQN replay/target-network machinery) "
        "were retained. Linear Q and SARSA share action-relative weights and therefore have symmetry sharing "
        "by construction even though they do not use stochastic D4 replay augmentation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    completion = RESULT_ROOT / "tier3_complete.json"
    if not completion.exists():
        raise FileNotFoundError("Tier 3 is incomplete")
    curve, curve_rows = summarize_curve()
    capabilities = summarize_capabilities()
    costs = summarize_costs()
    contrasts = paired_contrasts(curve)
    payload = {
        "training_seeds": list(SEEDS),
        "agents": list(AGENTS),
        "task4_curve": curve,
        "capabilities": capabilities,
        "costs": costs,
        "paired_contrasts": contrasts,
    }
    atomic_json(RESULT_ROOT / "summary.json", payload)
    write_csv(RESULT_ROOT / "task4_curve_by_seed.csv", curve_rows)
    write_csv(RESULT_ROOT / "costs.csv", costs)
    plot_curve(curve)
    plot_capability_profile(capabilities)
    plot_capability_delta(capabilities)
    plot_d4_effect(curve)
    (RESULT_ROOT / "REPORT.md").write_text(
        markdown_report(curve, capabilities, costs, contrasts), encoding="utf-8"
    )
    print(json.dumps({
        "summary": str((RESULT_ROOT / "summary.json").relative_to(ROOT)),
        "report": str((RESULT_ROOT / "REPORT.md").relative_to(ROOT)),
        "training_seeds": len(SEEDS),
        "models": len(AGENTS),
    }, indent=2))


if __name__ == "__main__":
    main()
