"""Run and summarize the frozen-weight E1 centre-bias calibration experiment.

Each run is one variant, lineup, and five-round seed cluster.  The 0.08 arm is
an evaluation-only runtime override of the frozen 0.04 agent; no checkpoint or
agent source is modified.  Decision traces are captured by the environment,
outside the measured agent callback time.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VARIANTS = {
    "center_00": ("dueling_hunt_center_0", 0.00),
    "center_04": ("dueling_hunt_center_04", 0.04),
    "center_08": ("dueling_hunt_center_04", 0.08),
    "center_12": ("dueling_hunt_center_12", 0.12),
}
LINEUPS = {
    "rule3": ["rule_based_agent"] * 3,
    "external_mixed": [
        "external_feature_is_everything",
        "external_deep_learning_killer",
        "rule_based_agent",
    ],
    "internal_mixed": [
        "frozen_ddqn_day7",
        "frozen_sarsa_best",
        "frozen_linear_q_final",
    ],
    "frozen_dddqn": [
        "dueling_ddqn_agent",
        "frozen_ddqn_day7",
        "frozen_ddqn_day11",
    ],
}
SEEDS = tuple(range(65001, 65021))
DIRECT_EXTENSION_SEEDS = tuple(range(65021, 65041))
DIRECT_EXTENSION_LINEUPS = {"external_mixed", "frozen_dddqn"}
DIRECT_EXTENSION_VARIANTS = {"center_04", "center_12"}
ROUNDS_PER_SEED = 5
RESULT_ROOT = ROOT / "results" / "e1_soft_center_bias"
PREVIOUS_PROTOCOL_SHA256S = {
    "9bbb55446dfece81c382f65d2222a7ab1600cd93be540d67a81d021a6aafb117",
    "37d9fc6a6aeb04aea78df4a16902c73e9ac61c0dedb3e1a31128ddbaa05f4e58",
    "27c0b58f4135dc3390a5a6503d8467376a2265cb98c12bfd0cf5ee2c8a99787e",
    "6045fc857c51009064733f1e2a2afbdba5faea643e897072482012a4a0c356b0",
}
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
MOVE_DELTAS = {
    "UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1),
    "LEFT": (-1, 0), "WAIT": (0, 0), "BOMB": (0, 0),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seeds_for(lineup: str, variant: str) -> tuple[int, ...]:
    if lineup in DIRECT_EXTENSION_LINEUPS and variant in DIRECT_EXTENSION_VARIANTS:
        return SEEDS + DIRECT_EXTENSION_SEEDS
    return SEEDS


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def center_distance(position) -> int:
    x, y = map(int, position)
    return max(3 - x, 0, x - 13) + max(3 - y, 0, y - 13)


def protocol() -> dict:
    files = []
    for agent in sorted({value[0] for value in VARIANTS.values()}):
        base = ROOT / "agent_code" / agent
        files.extend(base / name for name in ("callbacks.py", "features.py", "model.py", "dueling_ddqn_model.pt"))
    frozen_opponents = set(LINEUPS["internal_mixed"]) | set(LINEUPS["frozen_dddqn"])
    for agent in sorted(frozen_opponents):
        base = ROOT / "agent_code" / agent
        files.extend(path for path in base.rglob("*") if path.is_file() and path.suffix in {".py", ".pt", ".npz", ".json"})
    return {
        "experiment": "E1 soft hunt-phase centre-bias calibration",
        "scenario": "classic",
        "variants": {key: {"agent": value[0], "lambda": value[1]} for key, value in VARIANTS.items()},
        "lineups": LINEUPS,
        "seeds": list(SEEDS),
        "direct_extension": {
            "seeds": list(DIRECT_EXTENSION_SEEDS),
            "lineups": sorted(DIRECT_EXTENSION_LINEUPS),
            "variants": sorted(DIRECT_EXTENSION_VARIANTS),
            "question": "Direct paired comparison of lambda 0.12 versus 0.04",
        },
        "rounds_per_seed": ROUNDS_PER_SEED,
        "total_rounds": sum(
            len(seeds_for(lineup, variant)) * ROUNDS_PER_SEED
            for lineup in LINEUPS for variant in VARIANTS
        ),
        "training": False,
        "primary_metric": "mean_score",
        "pairing_unit": "seed cluster within lineup; five rounds aggregated before inference",
        "hunt_phase": "no coins, no crates, and at least one living opponent",
        "centre_box": {"x": [3, 13], "y": [3, 13]},
        "primary_mechanism_metrics": [
            "hunt_mean_center_distance", "hunt_center_progress_rate", "hunt_center_occupancy"
        ],
        "diagnostic_metrics": [
            "hunt_wait_rate", "hunt_revisit_rate", "hunt_two_cycle_rate",
            "hunt_decisions", "hunt_duration", "timeout_rate",
        ],
        "bootstrap": {"unit": "paired seed cluster", "replicates": 10000, "rng_seed": 65100},
        "expansion": {
            "base_seeds": list(range(65001, 65011)),
            "added_seeds": list(range(65011, 65021)),
            "added_lineups": ["internal_mixed", "frozen_dddqn"],
            "previous_protocol_sha256s": sorted(PREVIOUS_PROTOCOL_SHA256S),
            "note": "Append-only seed and lineup expansions; all earlier rounds are retained unchanged.",
        },
        "frozen_files": {str(path.relative_to(ROOT)): digest(path) for path in files},
        "lambda_08_note": "Evaluation-only override of HUNT_CENTER_Q_BIAS before setup; frozen 0.04 checkpoint and source otherwise unchanged.",
    }


def validate_frozen_files(spec: dict) -> None:
    for relative, expected in spec["frozen_files"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")


def raw_path(lineup: str, variant: str, seed: int) -> Path:
    return RESULT_ROOT / "raw_stats" / lineup / variant / f"seed_{seed}.json"


def trace_path(lineup: str, variant: str, seed: int) -> Path:
    return RESULT_ROOT / "traces" / lineup / variant / f"seed_{seed}.json"


def run_path(lineup: str, variant: str, seed: int) -> Path:
    return RESULT_ROOT / "manifests" / lineup / variant / f"seed_{seed}.run.json"


def run_one(variant: str, lineup: str, seed: int, rounds: int) -> None:
    import torch
    import main
    from environment import BombeRLeWorld

    spec = json.loads((RESULT_ROOT / "protocol.json").read_text(encoding="utf-8"))
    validate_frozen_files(spec)
    agent_code, bias = VARIANTS[variant]
    callbacks = importlib.import_module(f"agent_code.{agent_code}.callbacks")
    callbacks.HUNT_CENTER_Q_BIAS = bias
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    output = raw_path(lineup, variant, seed)
    trace_output = trace_path(lineup, variant, seed)
    manifest_output = run_path(lineup, variant, seed)
    for path in (output, trace_output, manifest_output):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    original_setup = BombeRLeWorld.setup_agents
    original_new = BombeRLeWorld.new_round
    original_action = BombeRLeWorld.perform_agent_action
    current = {"initial": [], "traces": [], "positions": []}

    def setup_agents(world, agents):
        original_setup(world, agents)
        world.agents[0].name = "candidate"
        world.agents[0].display_name = "candidate"

    def new_round(world):
        round_seed = seed * 100 + world.round
        world.rng = np.random.default_rng(round_seed)
        random.seed(round_seed)
        np.random.seed(round_seed)
        for index, agent in enumerate(world.agents):
            runtime = agent.backend.runner.fake_self
            if hasattr(runtime, "rng"):
                runtime.rng.seed(round_seed + index)
        original_new(world)
        current["positions"] = []
        initial = {
            "arena": world.arena.tolist(),
            "coins": [list(map(int, coin.get_state())) for coin in world.coins],
            "positions": [[int(agent.x), int(agent.y)] for agent in world.agents],
        }
        current["initial"].append({
            "round": int(world.round), "round_seed": round_seed,
            "sha256": hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest(),
        })

    def perform_agent_action(world, agent, action):
        if agent.name == "candidate":
            state = agent.last_game_state
            position = tuple(map(int, state["self"][-1]))
            field = state["field"]
            hunt = not state["coins"] and not np.any(field == 1) and bool(state["others"])
            dx, dy = MOVE_DELTAS.get(action, (0, 0))
            target = (position[0] + dx, position[1] + dy)
            before = center_distance(position)
            after = center_distance(target)
            history = current["positions"]
            current["traces"].append({
                "variant": variant, "lambda": bias, "lineup": lineup, "seed": seed,
                "round": int(world.round), "step": int(world.step),
                "x": position[0], "y": position[1], "action": action,
                "hunt_phase": bool(hunt), "center_distance": before,
                "inside_center_box": before == 0,
                "center_progress": bool(hunt and after < before),
                "center_progress_value": int(before - after) if hunt else 0,
                "wait": action == "WAIT",
                "revisit": bool(hunt and target in history[-32:]),
                "two_cycle": bool(hunt and len(history) >= 2 and target == history[-2]),
            })
            history.append(position)
        return original_action(world, agent, action)

    BombeRLeWorld.setup_agents = setup_agents
    BombeRLeWorld.new_round = new_round
    BombeRLeWorld.perform_agent_action = perform_agent_action
    command = [
        "main.py", "play", "--no-gui", "--agents", agent_code, *LINEUPS[lineup],
        "--scenario", "classic", "--n-rounds", str(rounds), "--seed", str(seed),
        "--save-stats", str(output), "--match-name", f"e1_{lineup}_{variant}_{seed}",
    ]
    main.main(command[1:])
    atomic_json(trace_output, current["traces"])
    manifest = {
        "variant": variant, "agent_code": agent_code, "lambda": bias, "lineup": lineup,
        "seed": seed, "rounds": rounds, "training": False, "command": command,
        "initial_conditions": current["initial"],
        "raw_stats": str(output.relative_to(ROOT)), "raw_sha256": digest(output),
        "trace": str(trace_output.relative_to(ROOT)), "trace_sha256": digest(trace_output),
        "protocol_sha256": digest(RESULT_ROOT / "protocol.json"),
    }
    atomic_json(manifest_output, manifest)
    validate_frozen_files(spec)


def validate_run(variant: str, lineup: str, seed: int, rounds: int) -> bool:
    output, trace, manifest = raw_path(lineup, variant, seed), trace_path(lineup, variant, seed), run_path(lineup, variant, seed)
    if not all(path.exists() for path in (output, trace, manifest)):
        return False
    data = json.loads(output.read_text(encoding="utf-8"))
    run = json.loads(manifest.read_text(encoding="utf-8"))
    return (
        len(data.get("by_round", {})) == rounds
        and int(data["by_agent"]["candidate"]["rounds"]) == rounds
        and run["raw_sha256"] == digest(output)
        and run["trace_sha256"] == digest(trace)
        and run["protocol_sha256"] in {
            digest(RESULT_ROOT / "protocol.json"),
            *PREVIOUS_PROTOCOL_SHA256S,
        }
    )


def round_rows(variant: str, lineup: str, seed: int) -> list[dict]:
    stats = json.loads(raw_path(lineup, variant, seed).read_text(encoding="utf-8"))
    traces = json.loads(trace_path(lineup, variant, seed).read_text(encoding="utf-8"))
    by_round_trace = defaultdict(list)
    for row in traces:
        by_round_trace[int(row["round"])].append(row)
    rows = []
    # Engine round keys include the match name and timestamp; insertion order
    # is the played round order and the trace stores the numeric round id.
    for round_id, round_data in enumerate(stats["by_round"].values(), start=1):
        outcome = round_data["agents"]["candidate"]
        hunt = [row for row in by_round_trace[round_id] if row["hunt_phase"]]
        hunt_steps = [row["step"] for row in hunt]
        n_hunt = len(hunt)
        rows.append({
            "variant": variant, "lambda": VARIANTS[variant][1], "lineup": lineup,
            "seed": seed, "round": round_id, "score": float(outcome["score"]),
            "coins": float(outcome["coins"]), "crates": float(outcome["crates"]),
            "bombs": float(outcome["bombs"]), "kills": float(outcome["kills"]),
            "suicides": float(outcome["suicides"]), "invalid": float(outcome["invalid"]),
            "survived": float(bool(outcome["survived"])),
            "action_ms": 1000.0 * float(outcome["time"]) / max(float(outcome["steps"]), 1.0),
            "hunt_decisions": n_hunt,
            "hunt_prevalence": n_hunt / max(len(by_round_trace[round_id]), 1),
            "hunt_mean_center_distance": float(np.mean([row["center_distance"] for row in hunt])) if hunt else np.nan,
            "hunt_center_progress_rate": float(np.mean([row["center_progress"] for row in hunt])) if hunt else np.nan,
            "hunt_center_occupancy": float(np.mean([row["inside_center_box"] for row in hunt])) if hunt else np.nan,
            "hunt_wait_rate": float(np.mean([row["wait"] for row in hunt])) if hunt else np.nan,
            "hunt_revisit_rate": float(np.mean([row["revisit"] for row in hunt])) if hunt else np.nan,
            "hunt_two_cycle_rate": float(np.mean([row["two_cycle"] for row in hunt])) if hunt else np.nan,
            "hunt_duration": (max(hunt_steps) - min(hunt_steps) + 1) if hunt_steps else 0,
            "timeout": float(int(round_data["steps"]) >= 400),
        })
    return rows


def nanmean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan")


def summarize() -> dict:
    rows = []
    for lineup in LINEUPS:
        for variant in VARIANTS:
            for seed in seeds_for(lineup, variant):
                if not validate_run(variant, lineup, seed, ROUNDS_PER_SEED):
                    raise ValueError(f"Missing or invalid run: {lineup}/{variant}/{seed}")
                rows.extend(round_rows(variant, lineup, seed))
    metrics = [key for key in rows[0] if key not in {"variant", "lambda", "lineup", "seed", "round"}]
    seed_rows = []
    for lineup in LINEUPS:
        for variant in VARIANTS:
            for seed in seeds_for(lineup, variant):
                group = [row for row in rows if row["lineup"] == lineup and row["variant"] == variant and row["seed"] == seed]
                seed_rows.append({"lineup": lineup, "variant": variant, "lambda": VARIANTS[variant][1], "seed": seed,
                                  **{metric: nanmean([row[metric] for row in group]) for metric in metrics}})
    aggregate = []
    for lineup in LINEUPS:
        for variant in VARIANTS:
            group = [row for row in seed_rows if row["lineup"] == lineup and row["variant"] == variant]
            aggregate.append({"lineup": lineup, "variant": variant, "lambda": VARIANTS[variant][1],
                              "seed_clusters": len(group), "rounds": len(group) * ROUNDS_PER_SEED,
                              **{metric: nanmean([row[metric] for row in group]) for metric in metrics}})
    comparisons = []
    rng = np.random.default_rng(65100)
    for lineup in LINEUPS:
        for left, right in (("center_04", "center_00"), ("center_04", "center_08"), ("center_04", "center_12")):
            for metric in (
                "score", "invalid", "suicides", "hunt_mean_center_distance",
                "hunt_center_progress_rate", "hunt_center_occupancy",
            ):
                diffs = []
                paired_seeds = sorted(set(seeds_for(lineup, left)) & set(seeds_for(lineup, right)))
                for seed in paired_seeds:
                    lrow = next(row for row in seed_rows if row["lineup"] == lineup and row["variant"] == left and row["seed"] == seed)
                    rrow = next(row for row in seed_rows if row["lineup"] == lineup and row["variant"] == right and row["seed"] == seed)
                    diffs.append(lrow[metric] - rrow[metric])
                diffs = np.asarray(diffs, dtype=float)
                samples = np.asarray([nanmean(diffs[rng.integers(0, len(diffs), len(diffs))]) for _ in range(10000)])
                comparisons.append({"lineup": lineup, "contrast": f"{left}-{right}", "metric": metric,
                                    "mean_difference": nanmean(diffs),
                                    "ci95": [float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))],
                                    "positive_seeds": int(np.sum(diffs > 0)), "nonmissing_seeds": int(np.sum(~np.isnan(diffs)))})
    payload = {"protocol": protocol(), "round_rows": rows, "seed_rows": seed_rows,
               "aggregate": aggregate, "paired_bootstrap": comparisons}
    atomic_json(RESULT_ROOT / "summary.json", payload)
    return payload


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--variant", choices=VARIANTS, required=True)
    run_parser.add_argument("--lineup", choices=LINEUPS, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--rounds", type=int, default=ROUNDS_PER_SEED)
    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "prepare":
        spec = protocol()
        target = RESULT_ROOT / "protocol.json"
        if target.exists() and json.loads(target.read_text(encoding="utf-8")) != spec:
            if digest(target) not in PREVIOUS_PROTOCOL_SHA256S:
                raise ValueError("Locked E1 protocol differs from the approved base protocol")
        atomic_json(target, spec)
    elif args.command == "run":
        if args.seed not in seeds_for(args.lineup, args.variant):
            raise ValueError("Seed is outside the locked protocol for this lineup and variant")
        run_one(args.variant, args.lineup, args.seed, args.rounds)
    else:
        summarize()


if __name__ == "__main__":
    main_cli()
