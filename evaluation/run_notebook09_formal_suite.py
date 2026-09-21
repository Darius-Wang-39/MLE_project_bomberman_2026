"""Run the locked benchmark and its two-candidate engine-aligned extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "notebook09_formal_mixed_benchmark"
REPORT_PATH = RESULT_ROOT / "report.json"
CONFIG_PATH = PROJECT_ROOT / "experiments" / "configs" / "notebook09_formal_mixed_benchmark.json"

LEGACY_CANDIDATES = (
    "dueling_hunt_center_04",
    "hunt_center_04_mix_0800",
    "dueling_ddqn_agent",
    "frozen_sarsa_best",
    "frozen_linear_q_final",
)
CANDIDATES = LEGACY_CANDIDATES + (
    "engine_aligned_0000",
    "frozen_engine_aligned_0200",
)
OPPONENTS = (
    "external_feature_is_everything",
    "external_deep_learning_killer",
    "rule_based_agent",
)
DEFAULT_SEEDS = (64001, 64002, 64003, 64004, 64005)
DEFAULT_ROUNDS = 30

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.summarize_stats import summarize, summarize_match


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol(rounds: int, seeds: tuple[int, ...]) -> dict:
    return {
        "experiment": "notebook09_formal_mixed_benchmark",
        "scenario": "classic",
        "rounds_per_seed": rounds,
        "seeds": list(seeds),
        "lineup_order": ["<candidate>", *OPPONENTS],
        "candidates": list(CANDIDATES),
        "opponents": list(OPPONENTS),
        "training": False,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_stats(path: Path, candidate: str, rounds: int) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload.get("by_round", {})) != rounds:
        raise ValueError(f"Wrong round count in {path}")
    matches = [name for name in payload.get("by_agent", {}) if name == candidate]
    if matches != [candidate]:
        raise ValueError(f"Candidate mismatch in {path}: {matches}")
    if int(payload["by_agent"][candidate].get("rounds", 0)) != rounds:
        raise ValueError(f"Candidate round count mismatch in {path}")


def command_for(candidate: str, seed: int, rounds: int, output: Path) -> list[str]:
    return [
        sys.executable,
        "main.py",
        "play",
        "--no-gui",
        "--agents",
        candidate,
        *OPPONENTS,
        "--scenario",
        "classic",
        "--n-rounds",
        str(rounds),
        "--seed",
        str(seed),
        "--save-stats",
        str(output),
    ]


def expected_sidecar(candidate: str, seed: int, rounds: int, output: Path, suite_protocol: dict) -> dict:
    command = command_for(candidate, seed, rounds, output)
    return {
        "candidate": candidate,
        "seed": seed,
        "rounds": rounds,
        "raw_stats": str(output.relative_to(PROJECT_ROOT)),
        "command": command[1:],
        "protocol": suite_protocol,
    }


def run_or_reuse(candidate: str, seed: int, rounds: int, suite_protocol: dict) -> dict:
    output = RESULT_ROOT / "raw" / candidate / f"seed_{seed}.json"
    sidecar = output.with_suffix(".run.json")
    expected = expected_sidecar(candidate, seed, rounds, output, suite_protocol)
    if output.exists():
        validate_stats(output, candidate, rounds)
        if not sidecar.exists():
            raise FileNotFoundError(f"Result exists without provenance sidecar: {sidecar}")
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        compatible_protocols = [suite_protocol]
        if candidate in LEGACY_CANDIDATES:
            legacy_protocol = {**suite_protocol, "candidates": list(LEGACY_CANDIDATES)}
            compatible_protocols.append(legacy_protocol)
        expected_variants = [
            {**expected, "protocol": compatible_protocol, "stats_sha256": sha256(output)}
            for compatible_protocol in compatible_protocols
        ]
        if recorded not in expected_variants:
            raise ValueError(f"Protocol or result drift: {sidecar}")
        print(f"REUSE {candidate} seed {seed}", flush=True)
        return {**expected, "status": "complete", "resumed": True}

    if sidecar.exists():
        raise FileExistsError(f"Orphan provenance sidecar: {sidecar}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = command_for(candidate, seed, rounds, output)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0 or not output.exists():
        raise RuntimeError(f"{candidate} seed {seed} failed:\n{completed.stdout[-8000:]}")
    validate_stats(output, candidate, rounds)
    atomic_json(sidecar, {**expected, "stats_sha256": sha256(output)})
    print(f"COMPLETE {candidate} seed {seed}", flush=True)
    return {**expected, "status": "complete", "resumed": False}


def aggregate_candidate(candidate: str, seeds: tuple[int, ...]) -> dict:
    paths = [RESULT_ROOT / "raw" / candidate / f"seed_{seed}.json" for seed in seeds]
    aggregate = {row["agent"]: row for row in summarize_match(paths)}
    by_seed = []
    for seed, path in zip(seeds, paths):
        metrics = {row["agent"]: row for row in summarize_match([path])}
        by_seed.append({
            "seed": seed,
            "candidate": metrics[candidate],
            "opponents": [metrics[name] for name in OPPONENTS],
        })
    return {
        "candidate": candidate,
        "aggregate": aggregate[candidate],
        "opponent_aggregate": [aggregate[name] for name in OPPONENTS],
        "by_seed": by_seed,
        "simple_metrics": summarize(paths, candidate),
    }


def run(rounds: int, seeds: tuple[int, ...]) -> dict:
    suite_protocol = protocol(rounds, seeds)
    if CONFIG_PATH.exists():
        existing = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if existing != suite_protocol:
            raise ValueError(f"Locked protocol drift: {CONFIG_PATH}")
    else:
        atomic_json(CONFIG_PATH, suite_protocol)

    runs = []
    for candidate in CANDIDATES:
        for seed in seeds:
            runs.append(run_or_reuse(candidate, seed, rounds, suite_protocol))
        progress = {"protocol": suite_protocol, "completed_runs": runs}
        atomic_json(RESULT_ROOT / "progress.json", progress)

    payload = {
        "protocol": suite_protocol,
        "runs": runs,
        "comparisons": [aggregate_candidate(candidate, seeds) for candidate in CANDIDATES],
    }
    atomic_json(REPORT_PATH, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()
    payload = run(args.rounds, tuple(args.seeds))
    print(json.dumps({
        "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        "completed_runs": len(payload["runs"]),
        "total_rounds": len(payload["runs"]) * args.rounds,
    }, indent=2))


if __name__ == "__main__":
    main()
