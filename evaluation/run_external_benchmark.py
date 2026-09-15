"""Evaluate read-only previous-cohort agents without mixing internal candidates.

The adapter names are prefixed with ``external_`` and all raw/output files are
kept below ``results/external_benchmark``. External source is never patched.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "external_benchmark"
AGENTS = (
    "baseline",
    "deep_learning_killer",
    "double_q",
    "dqn",
    "dqn_5tile",
    "feature_is_everything",
    "q_learning",
    "sarsa",
    "sarsa_lambda",
)
SMOKE_SEED = 61000
SCREEN_SEED = 61001
CONFIRM_SEEDS = (62001, 62002, 62003, 62004, 62005)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.summarize_stats import summarize


def external_name(name: str) -> str:
    return name if name.startswith("external_") else f"external_{name}"


def adapter_status(name: str) -> tuple[bool, str]:
    adapter = PROJECT_ROOT / "agent_code" / external_name(name)
    if not adapter.is_dir():
        return False, "external source/adapter missing"
    callback = adapter / "callbacks.py"
    if not callback.exists() or not callback.is_symlink():
        return False, "read-only callbacks link missing"
    if (adapter / "train.py").exists():
        return False, "unsafe adapter contains train.py"
    return True, "ready"


def play(name: str, rounds: int, seed: int, output: Path) -> dict:
    candidate = external_name(name)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "main.py",
        "play",
        "--no-gui",
        "--agents",
        candidate,
        "rule_based_agent",
        "rule_based_agent",
        "rule_based_agent",
        "--scenario",
        "classic",
        "--n-rounds",
        str(rounds),
        "--seed",
        str(seed),
        "--save-stats",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    record = {
        "agent": candidate,
        "rounds_requested": rounds,
        "seed": seed,
        "command": command[1:],
        "raw_stats": str(output.relative_to(PROJECT_ROOT)),
        "returncode": completed.returncode,
    }
    if completed.returncode == 0 and output.exists():
        record["status"] = "compatible"
        record["metrics"] = summarize([output], candidate)
    else:
        record["status"] = "incompatible"
        record["error"] = completed.stdout[-8000:]
    return record


def framework_preflight() -> None:
    """Fail once for shared framework/dependency errors before judging agents."""
    output = RESULT_ROOT / "preflight_rule_based.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "main.py", "play", "--no-gui", "--agents",
        "rule_based_agent", "rule_based_agent", "rule_based_agent", "rule_based_agent",
        "--scenario", "classic", "--n-rounds", "1", "--seed", str(SMOKE_SEED),
        "--save-stats", str(output),
    ]
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "Framework preflight failed; no external agent was classified.\n"
            + completed.stdout[-8000:]
        )


def run_smoke() -> dict:
    framework_preflight()
    records = []
    raw_dir = RESULT_ROOT / "smoke_raw"
    for name in AGENTS:
        ready, reason = adapter_status(name)
        if ready:
            record = play(name, 3, SMOKE_SEED, raw_dir / f"{name}.json")
        else:
            record = {"agent": external_name(name), "status": "incompatible", "reason": reason}
        records.append(record)
        print(f"{record['agent']}: {record['status']}", flush=True)
    payload = {"stage": "smoke", "seed": SMOKE_SEED, "rounds": 3, "agents": records}
    write_report("smoke.json", payload)
    return payload


def run_screen(resume: bool) -> dict:
    smoke = read_report("smoke.json")
    compatible = [row["agent"].removeprefix("external_") for row in smoke["agents"] if row["status"] == "compatible"]
    records = []
    raw_dir = RESULT_ROOT / "screen_raw"
    for name in compatible:
        raw_path = raw_dir / f"{name}.json"
        if resume and raw_path.exists():
            record = {
                "agent": external_name(name), "status": "compatible", "seed": SCREEN_SEED,
                "rounds_requested": 30, "raw_stats": str(raw_path.relative_to(PROJECT_ROOT)),
                "metrics": summarize([raw_path], external_name(name)), "resumed": True,
            }
        else:
            record = play(name, 30, SCREEN_SEED, raw_path)
        records.append(record)
        print(f"{record['agent']}: {record['status']}", flush=True)
    ranked = sorted(
        (row for row in records if row["status"] == "compatible"),
        key=lambda row: row["metrics"]["mean_score"], reverse=True,
    )
    payload = {
        "stage": "screen", "seed": SCREEN_SEED, "rounds": 30,
        "agents": records, "top_three": [row["agent"] for row in ranked[:3]],
    }
    write_report("screen.json", payload)
    return payload


def run_confirm(resume: bool) -> dict:
    screen = read_report("screen.json")
    records = []
    for candidate in screen["top_three"]:
        seed_rows = []
        name = candidate.removeprefix("external_")
        for seed in CONFIRM_SEEDS:
            raw_path = RESULT_ROOT / "confirm_raw" / name / f"seed_{seed}.json"
            if resume and raw_path.exists():
                row = {"seed": seed, "status": "compatible", "metrics": summarize([raw_path], candidate), "resumed": True}
            else:
                row = play(name, 30, seed, raw_path)
            seed_rows.append(row)
            print(f"{candidate} seed {seed}: {row['status']}", flush=True)
        successful = [row for row in seed_rows if row["status"] == "compatible"]
        combined_paths = [RESULT_ROOT / "confirm_raw" / name / f"seed_{row['seed']}.json" for row in successful]
        aggregate = summarize(combined_paths, candidate) if successful else None
        seed_scores = [row["metrics"]["mean_score"] for row in successful]
        records.append({
            "agent": candidate,
            "seeds": seed_rows,
            "aggregate": aggregate,
            "seed_score_stdev": statistics.stdev(seed_scores) if len(seed_scores) > 1 else 0.0,
        })
    payload = {"stage": "confirm", "seeds": CONFIRM_SEEDS, "rounds_per_seed": 30, "agents": records}
    write_report("confirm.json", payload)
    return payload


def write_report(filename: str, payload: dict) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_report(filename: str) -> dict:
    path = RESULT_ROOT / filename
    if not path.exists():
        raise SystemExit(f"Missing {path}; run the preceding stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("smoke", "screen", "confirm"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "smoke":
        payload = run_smoke()
    elif args.stage == "screen":
        payload = run_screen(args.resume)
    else:
        payload = run_confirm(args.resume)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
