"""Run Experiment 1: phase-gated versus ungated centre preference.

The candidate checkpoint, safety mask, objective routing, and all other
inference logic are frozen.  The ungated arm differs only by exposing the same
0.04 centre-progress term during non-emergency economic play.  Existing exact
no-centre and phase-gated runs from the calibration record are reused when
available; missing clusters and every ungated cluster are run here.
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

CANDIDATE = "dueling_hunt_center_04"
VARIANTS = {
    "no_center": {"lambda": 0.00, "economic_gate": True},
    "phase_gated": {"lambda": 0.04, "economic_gate": True},
    "ungated_center": {"lambda": 0.04, "economic_gate": False},
}
LINEUPS = {
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
SEEDS = tuple(range(65001, 65041))
ROUNDS_PER_SEED = 5
RESULT_ROOT = ROOT / "results" / "e1_phase_gating"
CALIBRATION_ROOT = ROOT / "results" / "e1_soft_center_bias"
CALIBRATION_VARIANTS = {"no_center": "center_00", "phase_gated": "center_04"}
PRE_BIAS_ONLY_PROTOCOL_SHA256 = "178f96082165827651fb8588468da65a05baf982b39af89653d0186e5a14c66d"
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
MOVE_DELTAS = {
    "UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1),
    "LEFT": (-1, 0), "WAIT": (0, 0), "BOMB": (0, 0),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def center_distance(position) -> int:
    x, y = map(int, position)
    return max(3 - x, 0, x - 13) + max(3 - y, 0, y - 13)


def frozen_files() -> dict[str, str]:
    agents = {CANDIDATE}
    for lineup in LINEUPS.values():
        agents.update(lineup)
    paths = []
    for agent in sorted(agents):
        base = ROOT / "agent_code" / agent
        paths.extend(
            path for path in base.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and "logs" not in path.parts
            and path.suffix.lower() in {".py", ".pt", ".npz", ".json"}
        )
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def protocol() -> dict:
    reused = {
        lineup: {
            variant: [seed for seed in SEEDS if calibration_valid(lineup, variant, seed)]
            for variant in CALIBRATION_VARIANTS
        }
        for lineup in LINEUPS
    }
    return {
        "experiment": "E1 phase-gating of a centre-directed heuristic",
        "question": (
            "Does phase-gating a centre-directed heuristic preserve economic progress "
            "and safety better than applying the same preference throughout play?"
        ),
        "scenario": "classic",
        "candidate": CANDIDATE,
        "variants": VARIANTS,
        "lineups": LINEUPS,
        "seeds": list(SEEDS),
        "rounds_per_seed": ROUNDS_PER_SEED,
        "total_rounds": len(LINEUPS) * len(VARIANTS) * len(SEEDS) * ROUNDS_PER_SEED,
        "training": False,
        "pairing_unit": "seed cluster within lineup; five rounds aggregated before inference",
        "implementation": {
            "no_center": "lambda=0; original safety mask and inference stack",
            "phase_gated": "lambda=.04; centre progress nonzero only in hunt phase",
            "ungated_center": (
                "lambda=.04; identical to phase_gated plus centre progress during "
                "non-emergency states with coins or crates remaining"
            ),
            "emergency": "candidate current-position danger score > 0",
            "hunt_phase": "no coins, no crates, and at least one living opponent",
            "centre_box": {"x": [3, 13], "y": [3, 13]},
        },
        "estimands": {
            "primary_contrast": "phase_gated - ungated_center",
            "reference_contrast": "phase_gated - no_center",
            "economy": ["coins", "crates"],
            "safety": ["suicides", "survived", "invalid"],
            "overall": ["score", "kills"],
            "mechanism": [
                "economic_center_progress_rate", "economic_center_occupancy",
                "hunt_center_progress_rate", "hunt_center_occupancy",
            ],
        },
        "interpretation_rule": (
            "Claim support only when paired effects consistently favor phase gating across "
            "lineups without an opposing economy or safety signal; otherwise describe the "
            "effect as lineup-specific, protective-only, or practically limited. Confidence "
            "intervals, not point estimates alone, determine clear directional evidence."
        ),
        "bootstrap": {"unit": "paired seed cluster", "replicates": 10000, "rng_seed": 65099},
        "seed_disclosure": (
            "Seeds 65001-65040 were already used in the calibration analysis. Exact overlapping "
            "arms are reused rather than rerun; this is a paired reanalysis plus a new ungated "
            "arm, not a fresh confirmatory holdout."
        ),
        "implementation_sha256": digest(Path(__file__)),
        "engine_files": {
            name: digest(ROOT / name)
            for name in ("main.py", "environment.py", "agents.py", "settings.py", "events.py", "items.py")
        },
        "unaffected_early_baseline_protocol_sha256": PRE_BIAS_ONLY_PROTOCOL_SHA256,
        "reused_calibration_clusters": reused,
        "frozen_files": frozen_files(),
    }


def validate_frozen_files(spec: dict) -> None:
    for relative, expected in spec["frozen_files"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")


def local_paths(lineup: str, variant: str, seed: int) -> tuple[Path, Path, Path]:
    return (
        RESULT_ROOT / "raw_stats" / lineup / variant / f"seed_{seed}.json",
        RESULT_ROOT / "traces" / lineup / variant / f"seed_{seed}.json",
        RESULT_ROOT / "manifests" / lineup / variant / f"seed_{seed}.run.json",
    )


def calibration_paths(lineup: str, variant: str, seed: int) -> tuple[Path, Path, Path]:
    old = CALIBRATION_VARIANTS[variant]
    return (
        CALIBRATION_ROOT / "raw_stats" / lineup / old / f"seed_{seed}.json",
        CALIBRATION_ROOT / "traces" / lineup / old / f"seed_{seed}.json",
        CALIBRATION_ROOT / "manifests" / lineup / old / f"seed_{seed}.run.json",
    )


def paths_for(lineup: str, variant: str, seed: int) -> tuple[Path, Path, Path, str]:
    local = local_paths(lineup, variant, seed)
    if validate_paths(*local, rounds=ROUNDS_PER_SEED, require_protocol=True):
        return *local, "phase_gating"
    if variant in CALIBRATION_VARIANTS:
        prior = calibration_paths(lineup, variant, seed)
        if calibration_valid(lineup, variant, seed):
            return *prior, "calibration"
    return *local, "missing"


def validate_paths(raw: Path, trace: Path, manifest: Path, rounds: int, require_protocol: bool) -> bool:
    if not all(path.exists() for path in (raw, trace, manifest)):
        return False
    try:
        data = json.loads(raw.read_text(encoding="utf-8"))
        run = json.loads(manifest.read_text(encoding="utf-8"))
        valid = (
            len(data.get("by_round", {})) == rounds
            and int(data["by_agent"]["candidate"]["rounds"]) == rounds
            and run["raw_sha256"] == digest(raw)
            and run["trace_sha256"] == digest(trace)
        )
        if require_protocol:
            valid = valid and run["protocol_sha256"] in {
                digest(RESULT_ROOT / "protocol.json"), PRE_BIAS_ONLY_PROTOCOL_SHA256,
            }
        return bool(valid)
    except (KeyError, ValueError, json.JSONDecodeError):
        return False


def calibration_valid(lineup: str, variant: str, seed: int) -> bool:
    if variant not in CALIBRATION_VARIANTS:
        return False
    return validate_paths(
        *calibration_paths(lineup, variant, seed),
        rounds=ROUNDS_PER_SEED,
        require_protocol=False,
    )


def make_ungated_act(callbacks, torch):
    """Return the frozen callback with only the explicit centre-Q term ungated.

    The feature matrix passed to the neural network is left byte-for-byte
    unchanged.  This prevents the learned weights on the hunt-only feature from
    becoming an additional treatment in economic states.
    """

    def act(self, game_state):
        allowed = self.feature_extractor.safe_action_mask(game_state)
        allowed_indices = np.flatnonzero(allowed)
        if allowed_indices.size == 0:
            return "WAIT"
        action_features = self.feature_extractor.action_features(game_state)
        position = tuple(game_state["self"][-1])
        if self.position_round != game_state["round"]:
            self.recent_positions.clear()
            self.position_round = game_state["round"]
        revisit_penalties = np.zeros(len(ACTIONS), dtype=np.float32)
        danger_index = callbacks.FEATURE_NAMES.index("target_danger")
        hunt_phase = (
            not game_state["coins"]
            and not np.any(game_state["field"] == 1)
            and bool(game_state["others"])
        )
        if hunt_phase or not allowed[ACTIONS.index("BOMB")]:
            for index, action in enumerate(ACTIONS):
                if action == "BOMB" or action_features[index, danger_index] > 0.0:
                    continue
                target = (
                    position[0] + MOVE_DELTAS[action][0],
                    position[1] + MOVE_DELTAS[action][1],
                )
                revisit_penalties[index] = callbacks.REVISIT_Q_PENALTY * min(
                    self.recent_positions.count(target), 3
                )
        self.recent_positions.append(position)

        if self.train and self.rng.random() < callbacks.exploration_rate(self):
            return ACTIONS[self.rng.choice(allowed_indices.tolist())]

        state = torch.from_numpy(action_features.astype(np.float32).reshape(-1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze(0).cpu().numpy()
        center_index = callbacks.FEATURE_NAMES.index("hunt_center_progress")
        objective_index = callbacks.FEATURE_NAMES.index("objective_target")
        center_progress = action_features[:, center_index].copy()
        economic = bool(game_state["coins"]) or bool(np.any(game_state["field"] == 1))
        non_emergency = self.feature_extractor._danger_score(game_state, position) == 0.0
        if economic and non_emergency:
            before = self.feature_extractor.center_box_distance(position)
            legal = self.feature_extractor.legal_action_mask(game_state)
            for index, action in enumerate(ACTIONS):
                target = (
                    position[0] + MOVE_DELTAS[action][0],
                    position[1] + MOVE_DELTAS[action][1],
                )
                center_progress[index] = (
                    float(before - self.feature_extractor.center_box_distance(target))
                    if legal[index]
                    else 0.0
                )
        q_values = (
            q_values
            + self.hunt_center_q_bias * center_progress
            + callbacks.OBJECTIVE_Q_BIAS * action_features[:, objective_index]
            - revisit_penalties
        )
        masked_q = np.where(allowed, q_values, -np.inf)
        best = np.flatnonzero(np.isclose(masked_q, np.max(masked_q)))
        return ACTIONS[self.rng.choice(best.tolist())]

    return act


def run_one(variant: str, lineup: str, seed: int) -> None:
    import torch
    import main
    from environment import BombeRLeWorld

    spec = json.loads((RESULT_ROOT / "protocol.json").read_text(encoding="utf-8"))
    validate_frozen_files(spec)
    callbacks = importlib.import_module(f"agent_code.{CANDIDATE}.callbacks")
    callbacks.HUNT_CENTER_Q_BIAS = VARIANTS[variant]["lambda"]
    if variant == "ungated_center":
        callbacks.act = make_ungated_act(callbacks, torch)
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    raw, trace, manifest = local_paths(lineup, variant, seed)
    if any(path.exists() for path in (raw, trace, manifest)):
        raise FileExistsError(f"Partial or existing run at {manifest.parent}")
    for path in (raw, trace, manifest):
        path.parent.mkdir(parents=True, exist_ok=True)

    original_setup = BombeRLeWorld.setup_agents
    original_new = BombeRLeWorld.new_round
    original_action = BombeRLeWorld.perform_agent_action
    current = {"initial": [], "traces": [], "positions": [], "extractor": None}

    def setup_agents(world, agents):
        original_setup(world, agents)
        candidate = world.agents[0]
        candidate.name = "candidate"
        candidate.display_name = "candidate"
        runtime = candidate.backend.runner.fake_self
        current["extractor"] = runtime.feature_extractor

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
            danger = float(current["extractor"]._danger_score(state, position))
            economic = (bool(state["coins"]) or bool(np.any(field == 1))) and danger == 0.0
            dx, dy = MOVE_DELTAS.get(action, (0, 0))
            target = (position[0] + dx, position[1] + dy)
            before, after = center_distance(position), center_distance(target)
            history = current["positions"]
            current["traces"].append({
                "variant": variant, "lineup": lineup, "seed": seed,
                "round": int(world.round), "step": int(world.step),
                "x": position[0], "y": position[1], "action": action,
                "hunt_phase": bool(hunt), "economic_nonemergency": bool(economic),
                "emergency": bool(danger > 0.0), "center_distance": before,
                "inside_center_box": before == 0,
                "center_progress": bool(after < before),
                "center_progress_value": int(before - after),
                "wait": action == "WAIT",
                "revisit": bool(target in history[-32:]),
                "two_cycle": bool(len(history) >= 2 and target == history[-2]),
            })
            history.append(position)
        return original_action(world, agent, action)

    BombeRLeWorld.setup_agents = setup_agents
    BombeRLeWorld.new_round = new_round
    BombeRLeWorld.perform_agent_action = perform_agent_action
    command = [
        "main.py", "play", "--no-gui", "--agents", CANDIDATE, *LINEUPS[lineup],
        "--scenario", "classic", "--n-rounds", str(ROUNDS_PER_SEED), "--seed", str(seed),
        "--save-stats", str(raw), "--match-name", f"e1_gate_{lineup}_{variant}_{seed}",
    ]
    main.main(command[1:])
    atomic_json(trace, current["traces"])
    atomic_json(manifest, {
        "variant": variant, "lineup": lineup, "seed": seed,
        "rounds": ROUNDS_PER_SEED, "training": False, "command": command,
        "initial_conditions": current["initial"],
        "raw_stats": str(raw.relative_to(ROOT)), "raw_sha256": digest(raw),
        "trace": str(trace.relative_to(ROOT)), "trace_sha256": digest(trace),
        "protocol_sha256": digest(RESULT_ROOT / "protocol.json"),
    })
    validate_frozen_files(spec)


def round_rows(lineup: str, variant: str, seed: int) -> list[dict]:
    raw, trace, _, source = paths_for(lineup, variant, seed)
    stats = json.loads(raw.read_text(encoding="utf-8"))
    traces = json.loads(trace.read_text(encoding="utf-8"))
    by_round_trace = defaultdict(list)
    for row in traces:
        by_round_trace[int(row["round"])].append(row)
    rows = []
    for round_id, round_data in enumerate(stats["by_round"].values(), start=1):
        outcome = round_data["agents"]["candidate"]
        decisions = by_round_trace[round_id]
        hunt = [row for row in decisions if row["hunt_phase"]]
        if source == "calibration":
            economic = []
        else:
            economic = [row for row in decisions if row["economic_nonemergency"]]

        def phase_mean(group, key):
            return float(np.mean([row[key] for row in group])) if group else np.nan

        rows.append({
            "variant": variant, "lineup": lineup, "seed": seed, "round": round_id,
            "source": source, "score": float(outcome["score"]),
            "coins": float(outcome["coins"]), "crates": float(outcome["crates"]),
            "bombs": float(outcome["bombs"]), "kills": float(outcome["kills"]),
            "suicides": float(outcome["suicides"]), "invalid": float(outcome["invalid"]),
            "survived": float(bool(outcome["survived"])),
            "action_ms": 1000.0 * float(outcome["time"]) / max(float(outcome["steps"]), 1.0),
            "timeout": float(int(round_data["steps"]) >= 400),
            "hunt_decisions": len(hunt),
            "hunt_center_distance": phase_mean(hunt, "center_distance"),
            "hunt_center_progress_rate": phase_mean(hunt, "center_progress"),
            "hunt_center_occupancy": phase_mean(hunt, "inside_center_box"),
            "economic_decisions": len(economic) if source != "calibration" else np.nan,
            "economic_center_distance": phase_mean(economic, "center_distance"),
            "economic_center_progress_rate": phase_mean(economic, "center_progress"),
            "economic_center_occupancy": phase_mean(economic, "inside_center_box"),
        })
    return rows


def nanmean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan")


def summarize() -> dict:
    spec = json.loads((RESULT_ROOT / "protocol.json").read_text(encoding="utf-8"))
    validate_frozen_files(spec)
    rows = []
    ledger = []
    for lineup in LINEUPS:
        for variant in VARIANTS:
            for seed in SEEDS:
                raw, trace, manifest, source = paths_for(lineup, variant, seed)
                if source == "missing":
                    raise ValueError(f"Missing run: {lineup}/{variant}/{seed}")
                ledger.append({"lineup": lineup, "variant": variant, "seed": seed, "source": source,
                               "raw_sha256": digest(raw), "trace_sha256": digest(trace),
                               "manifest": str(manifest.relative_to(ROOT))})
                rows.extend(round_rows(lineup, variant, seed))

    metric_names = [
        key for key in rows[0]
        if key not in {"variant", "lineup", "seed", "round", "source"}
    ]
    seed_rows = []
    for lineup in LINEUPS:
        for variant in VARIANTS:
            for seed in SEEDS:
                group = [row for row in rows if row["lineup"] == lineup and row["variant"] == variant and row["seed"] == seed]
                seed_rows.append({
                    "lineup": lineup, "variant": variant, "seed": seed,
                    **{metric: nanmean([row[metric] for row in group]) for metric in metric_names},
                })

    aggregate = []
    for lineup in LINEUPS:
        for variant in VARIANTS:
            group = [row for row in seed_rows if row["lineup"] == lineup and row["variant"] == variant]
            aggregate.append({
                "lineup": lineup, "variant": variant,
                "seed_clusters": len(group), "rounds": len(group) * ROUNDS_PER_SEED,
                **{metric: nanmean([row[metric] for row in group]) for metric in metric_names},
            })

    comparison_metrics = [
        "score", "coins", "crates", "kills", "suicides", "invalid", "survived",
        "hunt_center_distance", "hunt_center_progress_rate", "hunt_center_occupancy",
    ]
    comparisons = []
    rng = np.random.default_rng(spec["bootstrap"]["rng_seed"])
    for lineup in LINEUPS:
        for left, right in (("phase_gated", "ungated_center"), ("phase_gated", "no_center"), ("ungated_center", "no_center")):
            for metric in comparison_metrics:
                diffs = []
                for seed in SEEDS:
                    left_row = next(row for row in seed_rows if row["lineup"] == lineup and row["variant"] == left and row["seed"] == seed)
                    right_row = next(row for row in seed_rows if row["lineup"] == lineup and row["variant"] == right and row["seed"] == seed)
                    diffs.append(left_row[metric] - right_row[metric])
                diffs = np.asarray(diffs, dtype=float)
                observed = diffs[~np.isnan(diffs)]
                samples = np.asarray([
                    float(np.mean(observed[rng.integers(0, len(observed), len(observed))]))
                    for _ in range(spec["bootstrap"]["replicates"])
                ])
                comparisons.append({
                    "lineup": lineup, "contrast": f"{left}-{right}", "metric": metric,
                    "mean_difference": float(np.mean(observed)),
                    "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
                    "positive_seeds": int(np.sum(observed > 0)),
                    "zero_seeds": int(np.sum(observed == 0)), "paired_seeds": int(len(observed)),
                })

    payload = {
        "protocol": spec, "ledger": ledger, "round_rows": rows,
        "seed_rows": seed_rows, "aggregate": aggregate, "paired_bootstrap": comparisons,
    }
    atomic_json(RESULT_ROOT / "summary.json", payload)
    return payload


def status() -> dict:
    counts = defaultdict(int)
    missing = []
    for lineup in LINEUPS:
        for variant in VARIANTS:
            for seed in SEEDS:
                source = paths_for(lineup, variant, seed)[3]
                counts[source] += 1
                if source == "missing":
                    missing.append({"lineup": lineup, "variant": variant, "seed": seed})
    return {"counts": dict(counts), "missing": missing, "total_clusters": len(LINEUPS) * len(VARIANTS) * len(SEEDS)}


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--variant", choices=VARIANTS, required=True)
    run_parser.add_argument("--lineup", choices=LINEUPS, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    sub.add_parser("summarize")
    sub.add_parser("status")
    args = parser.parse_args()

    if args.command == "prepare":
        target = RESULT_ROOT / "protocol.json"
        spec = protocol()
        if target.exists() and json.loads(target.read_text(encoding="utf-8")) != spec:
            if digest(target) != PRE_BIAS_ONLY_PROTOCOL_SHA256:
                raise ValueError("Locked E1 phase-gating protocol differs from current inputs")
        atomic_json(target, spec)
    elif args.command == "run":
        if args.seed not in SEEDS:
            raise ValueError("Seed is outside the locked protocol")
        if paths_for(args.lineup, args.variant, args.seed)[3] != "missing":
            raise FileExistsError("A valid local or calibration run already exists")
        run_one(args.variant, args.lineup, args.seed)
    elif args.command == "summarize":
        summarize()
    else:
        print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    main_cli()
