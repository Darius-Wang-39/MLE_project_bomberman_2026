"""Run paired candidate-agent matches against three fixed external benchmarks."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "external_benchmark" / "internal_challenge"
INTERNAL_AGENTS = (
    "dueling_ddqn_agent",
    "dueling_hunt_center_04",
    "frozen_ddqn_day7",
    "hunt_center_04_mix_0800",
    "external_deep_learning_killer",
)
EXTERNAL_AGENTS = ("external_feature_is_everything", "external_sarsa", "external_q_learning")
DEFAULT_SEEDS = (63001, 63002, 63003, 63004, 63005)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.summarize_stats import summarize, summarize_match


def command_for(candidate: str, seed: int, rounds: int, output: Path) -> list[str]:
    return [
        sys.executable, "main.py", "play", "--no-gui", "--agents",
        candidate, *EXTERNAL_AGENTS,
        "--scenario", "classic", "--n-rounds", str(rounds),
        "--seed", str(seed), "--save-stats", str(output),
    ]


def run_one(candidate: str, seed: int, rounds: int, output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = command_for(candidate, seed, rounds, output)
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    record = {
        "candidate": candidate,
        "seed": seed,
        "rounds_requested": rounds,
        "raw_stats": str(output.relative_to(PROJECT_ROOT)),
        "command": command[1:],
        "returncode": completed.returncode,
    }
    if completed.returncode == 0 and output.exists():
        record["status"] = "complete"
        record["candidate_metrics"] = summarize([output], candidate)
    else:
        record["status"] = "failed"
        record["error"] = completed.stdout[-8000:]
    return record


def aggregate_candidate(candidate: str, completed: list[dict]) -> dict:
    paths = [PROJECT_ROOT / row["raw_stats"] for row in completed]
    all_agents = {row["agent"]: row for row in summarize_match(paths)}
    seed_rows = []
    for row, path in zip(completed, paths):
        metrics = {item["agent"]: item for item in summarize_match([path])}
        seed_rows.append({
            "seed": row["seed"],
            "agents": [metrics[name] for name in (candidate, *EXTERNAL_AGENTS)],
        })
    scores = [row["candidate_metrics"]["mean_score"] for row in completed]
    return {
        "candidate": candidate,
        "completed_seeds": len(completed),
        "candidate_seed_score_stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "aggregate_agents": [all_agents[name] for name in (candidate, *EXTERNAL_AGENTS)],
        "by_seed": seed_rows,
    }


def run(rounds: int, seeds: tuple[int, ...], resume: bool) -> dict:
    runs = []
    comparisons = []
    for candidate in INTERNAL_AGENTS:
        completed = []
        for seed in seeds:
            output = RESULT_ROOT / "raw" / candidate / f"seed_{seed}.json"
            if resume and output.exists():
                row = {
                    "candidate": candidate,
                    "seed": seed,
                    "rounds_requested": rounds,
                    "raw_stats": str(output.relative_to(PROJECT_ROOT)),
                    "status": "complete",
                    "candidate_metrics": summarize([output], candidate),
                    "resumed": True,
                }
            else:
                row = run_one(candidate, seed, rounds, output)
            runs.append(row)
            print(f"{candidate} seed {seed}: {row['status']}", flush=True)
            if row["status"] == "complete":
                completed.append(row)
        comparisons.append(aggregate_candidate(candidate, completed) if completed else {
            "candidate": candidate, "completed_seeds": 0,
        })

    payload = {
        "experiment": "internal_vs_fixed_external_benchmarks",
        "scenario": "classic",
        "rounds_per_seed": rounds,
        "seeds": list(seeds),
        "lineup_order": ["<internal_candidate>", *EXTERNAL_AGENTS],
        "internal_agents": list(INTERNAL_AGENTS),
        "external_benchmarks": list(EXTERNAL_AGENTS),
        "runs": runs,
        "comparisons": comparisons,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    payload = run(args.rounds, tuple(args.seeds), args.resume)
    print(json.dumps({
        "report": str((RESULT_ROOT / "report.json").relative_to(PROJECT_ROOT)),
        "comparisons": payload["comparisons"],
    }, indent=2))


if __name__ == "__main__":
    main()
