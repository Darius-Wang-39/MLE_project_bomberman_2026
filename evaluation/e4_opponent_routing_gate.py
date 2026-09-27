"""Run E4: opponent-routing gate ablation.

The frozen ``dueling_hunt_center_04`` policy, safety layer, economic routing,
hunt-centre bias, and revisit penalty are shared by all arms.  Only the
``objective_target`` column produced by an opponent objective is gated before
network inference and the direct +1.0 objective Q bonus.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import e3_economic_routing as harness


CANDIDATE = "dueling_hunt_center_04"
VARIANTS = ("opponent_route_off", "fallback_final", "strict_hunt_only")
LINEUPS = {
    "external_mixed": [
        "external_feature_is_everything",
        "external_deep_learning_killer",
        "rule_based_agent",
    ],
    "frozen_dddqn": [
        "dueling_ddqn_agent", "frozen_ddqn_day7", "frozen_ddqn_day11",
    ],
}
SEEDS = tuple(range(68001, 68041))
ROUNDS_PER_SEED = 5
RESULT_ROOT = ROOT / "results" / "e4_opponent_routing_gate"
PROTOCOL_PATH = RESULT_ROOT / "protocol.json"
SUMMARY_PATH = RESULT_ROOT / "summary.json"
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")


digest = harness.digest
atomic_json = harness.atomic_json
classify_objective_source = harness.classify_objective_source


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
    return {
        "experiment": "E4 opponent-routing gate ablation",
        "question": (
            "Does fallback opponent routing improve offensive conversion, and "
            "does a strict hunt-only gate better protect remaining economic opportunities?"
        ),
        "candidate": CANDIDATE,
        "checkpoint_sha256": digest(
            ROOT / "agent_code" / CANDIDATE / "dueling_ddqn_model.pt"
        ),
        "scenario": "classic",
        "variants": {
            "opponent_route_off": (
                "clear objective_target when its source is opponent; retain learned "
                "opponent_progress, bomb_opponent_yield, bomb_trap_pressure, and "
                "armed_opponent_threat features"
            ),
            "fallback_final": (
                "retain current opponent fallback after coin/crate/escape routing "
                "does not select an objective and no crates remain"
            ),
            "strict_hunt_only": (
                "retain an opponent objective only when coins are empty, crates are "
                "absent, and at least one opponent remains"
            ),
        },
        "lineups": LINEUPS,
        "seeds": list(SEEDS),
        "seed_status": "fresh; disjoint from E1-E3 formal seeds",
        "rounds_per_seed": ROUNDS_PER_SEED,
        "total_rounds": len(VARIANTS) * len(LINEUPS) * len(SEEDS) * ROUNDS_PER_SEED,
        "training": False,
        "exploration": 0.0,
        "pairing_unit": "lineup/seed cluster; average five rounds before inference",
        "shared_layers": [
            "network weights and 132-dimensional architecture",
            "six-step safe_action_mask", "hybrid two-branch bomb escape rule",
            "escape and economic objective routing", "hunt-centre Q bias 0.04",
            "revisit penalty 0.60", "all learned opponent features",
        ],
        "manipulation": (
            "source-selective zeroing of opponent objective_target before both "
            "network inference and the +1.0 direct objective Q bonus"
        ),
        "objective_sources": [
            "escape", "coin", "immediate_crate_bomb",
            "productive_crate_target", "opponent", "none",
        ],
        "prespecified_contrasts": [
            "fallback_final-opponent_route_off",
            "strict_hunt_only-fallback_final",
            "strict_hunt_only-opponent_route_off",
        ],
        "primary_metric": "mean_score",
        "metric_families": {
            "offensive_conversion": [
                "kills", "opponent_route_opportunities",
                "opponent_route_opportunity_rate", "opponent_target_selection_rate",
                "opponent_progress", "trap_pressure_bombs",
                "nonzero_trap_pressure_bomb_rate", "bomb_opponent_yield",
            ],
            "economic_opportunity_cost": [
                "coins", "crates", "coin_routing_activations",
                "coin_routing_activation_rate", "opponent_routes_with_coin",
                "opponent_with_coin_rate",
            ],
            "safety": [
                "suicides", "survived", "invalid", "self_trapped_after_bomb",
            ],
            "exposure": ["hunt_steps", "hunt_round", "opponent_route_opportunities"],
        },
        "bootstrap": {
            "unit": "paired seed cluster", "replicates": 10000, "rng_seed": 68099,
        },
        "implementation_sha256": digest(Path(__file__)),
        "engine_files": {
            name: digest(ROOT / name)
            for name in (
                "main.py", "environment.py", "agents.py", "settings.py",
                "events.py", "items.py",
            )
        },
        "frozen_files": frozen_files(),
    }


def validate_frozen_files(spec: dict) -> None:
    for relative, expected in spec["frozen_files"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")


def paths(lineup: str, variant: str, seed: int):
    base = RESULT_ROOT / "runs" / lineup / variant
    return (
        base / f"seed_{seed}.stats.json",
        base / f"seed_{seed}.trace.json",
        base / f"seed_{seed}.run.json",
    )


def hunt_phase(game_state: dict) -> bool:
    return bool(
        not game_state["coins"]
        and not np.any(game_state["field"] == 1)
        and game_state["others"]
    )


def opponent_route_enabled(variant: str, source: str, game_state: dict) -> bool:
    if source != "opponent":
        return True
    if variant == "opponent_route_off":
        return False
    if variant == "fallback_final":
        return True
    if variant == "strict_hunt_only":
        return hunt_phase(game_state)
    raise ValueError(f"Unknown E4 variant: {variant}")


def make_act(callbacks, torch, variant: str, trace_rows: list):
    def act(self, game_state: dict) -> str:
        allowed = self.feature_extractor.safe_action_mask(game_state)
        allowed_indices = np.flatnonzero(allowed)
        if allowed_indices.size == 0:
            return "WAIT"

        original_features = self.feature_extractor.action_features(game_state)
        source = classify_objective_source(
            self.feature_extractor, game_state, original_features
        )
        objective_index = callbacks.FEATURE_NAMES.index("objective_target")
        original_objective = original_features[:, objective_index] > 0.5
        enabled = opponent_route_enabled(variant, source, game_state)
        action_features = original_features.copy()
        if not enabled:
            action_features[:, objective_index] = 0.0

        position = tuple(game_state["self"][-1])
        if self.position_round != game_state["round"]:
            self.recent_positions.clear()
            self.position_round = game_state["round"]
        revisit_penalties = np.zeros(len(ACTIONS), dtype=np.float32)
        danger_index = callbacks.FEATURE_NAMES.index("target_danger")
        is_hunt = hunt_phase(game_state)
        if is_hunt or not allowed[ACTIONS.index("BOMB")]:
            for index, action in enumerate(ACTIONS):
                if action == "BOMB" or action_features[index, danger_index] > 0.0:
                    continue
                target = (
                    position[0] + callbacks.MOVE_DELTAS[action][0],
                    position[1] + callbacks.MOVE_DELTAS[action][1],
                )
                revisit_penalties[index] = callbacks.REVISIT_Q_PENALTY * min(
                    self.recent_positions.count(target), 3
                )
        self.recent_positions.append(position)

        state = torch.from_numpy(
            action_features.astype(np.float32).reshape(-1)
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            network_q = self.policy_net(state).squeeze(0).cpu().numpy()
        center_index = callbacks.FEATURE_NAMES.index("hunt_center_progress")
        final_q = (
            network_q
            + self.hunt_center_q_bias * action_features[:, center_index]
            + callbacks.OBJECTIVE_Q_BIAS * action_features[:, objective_index]
            - revisit_penalties
        )
        masked_q = np.where(allowed, final_q, -np.inf)
        best = np.flatnonzero(np.isclose(masked_q, np.max(masked_q)))
        chosen_index = int(self.rng.choice(best.tolist()))
        chosen = ACTIONS[chosen_index]
        opportunity = bool(source == "opponent" and original_objective.any())
        trace_rows.append({
            "round": int(game_state["round"]),
            "step": int(game_state["step"]),
            "variant": variant,
            "action": chosen,
            "objective_source": source,
            "objective_enabled": bool(enabled),
            "opponent_route_opportunity": opportunity,
            "opponent_route_enabled": bool(opportunity and enabled),
            "objective_target_selected": bool(original_objective[chosen_index]),
            "opponent_progress": float(original_features[
                chosen_index, callbacks.FEATURE_NAMES.index("opponent_progress")
            ]),
            "bomb_opponent_yield": float(original_features[
                chosen_index, callbacks.FEATURE_NAMES.index("bomb_opponent_yield")
            ]),
            "bomb_trap_pressure": float(original_features[
                chosen_index, callbacks.FEATURE_NAMES.index("bomb_trap_pressure")
            ]),
            "coin_routing_activation": bool(
                source == "coin" and original_objective.any()
            ),
            "coins_present": bool(game_state["coins"]),
            "safe_productive_bomb": bool(original_features[
                chosen_index, callbacks.FEATURE_NAMES.index("safe_productive_bomb")
            ] > 0.5),
            "hunt_phase": bool(is_hunt),
            "decision_ms": None,
        })
        return chosen
    return act


def configure_harness() -> None:
    harness.CANDIDATE = CANDIDATE
    harness.VARIANTS = VARIANTS
    harness.LINEUPS = LINEUPS
    harness.SEEDS = SEEDS
    harness.ROUNDS_PER_SEED = ROUNDS_PER_SEED
    harness.RESULT_ROOT = RESULT_ROOT
    harness.PROTOCOL_PATH = PROTOCOL_PATH
    harness.SUMMARY_PATH = SUMMARY_PATH
    harness.make_act = make_act
    harness.validate_frozen_files = validate_frozen_files


def run_one(variant: str, lineup: str, seed: int) -> None:
    configure_harness()
    harness.run_one(variant, lineup, seed)


def round_rows(lineup: str, variant: str, seed: int) -> list[dict]:
    raw_path, trace_path, _ = paths(lineup, variant, seed)
    stats = json.loads(raw_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    decisions = defaultdict(list)
    bombs = defaultdict(list)
    for row in trace["decisions"]:
        decisions[int(row["round"])].append(row)
    for row in trace["bombs"]:
        bombs[int(row["round"])].append(row)
    rows = []
    for round_id, round_data in enumerate(stats["by_round"].values(), start=1):
        outcome = round_data["agents"]["candidate"]
        rd, rb = decisions[round_id], bombs[round_id]
        opportunities = [row for row in rd if row["opponent_route_opportunity"]]
        enabled = [row for row in opportunities if row["opponent_route_enabled"]]
        coin_routes = [row for row in rd if row["coin_routing_activation"]]
        chosen_bombs = [row for row in rd if row["action"] == "BOMB"]
        pressure_bombs = [
            row for row in chosen_bombs if row["bomb_trap_pressure"] > 0.0
        ]
        times = [
            float(row["decision_ms"]) for row in rd
            if row["decision_ms"] is not None
        ]
        rows.append({
            "variant": variant, "lineup": lineup, "seed": seed, "round": round_id,
            "score": float(outcome["score"]), "coins": float(outcome["coins"]),
            "crates": float(outcome["crates"]), "bombs": float(outcome["bombs"]),
            "kills": float(outcome["kills"]), "suicides": float(outcome["suicides"]),
            "invalid": float(outcome["invalid"]),
            "survived": float(bool(outcome["survived"])),
            "decision_steps": float(len(rd)),
            "opponent_route_opportunities": float(len(opportunities)),
            "opponent_route_enabled_steps": float(len(enabled)),
            "opponent_target_selections": float(sum(
                row["objective_target_selected"] for row in opportunities
            )),
            "opponent_progress": float(np.mean([
                row["opponent_progress"] for row in opportunities
            ])) if opportunities else math.nan,
            "trap_pressure_bombs": float(len(pressure_bombs)),
            "selected_bombs": float(len(chosen_bombs)),
            "bomb_opponent_yield": float(sum(
                row["bomb_opponent_yield"] for row in chosen_bombs
            )),
            "coin_routing_activations": float(len(coin_routes)),
            "opponent_routes_with_coin": float(sum(
                row["coins_present"] for row in opportunities
            )),
            "hunt_steps": float(sum(row["hunt_phase"] for row in rd)),
            "hunt_round": float(any(row["hunt_phase"] for row in rd)),
            "self_trapped_after_bomb": float(sum(
                row["self_trapped_after_bomb"] for row in rb
            )),
            "decision_ms_mean": float(np.mean(times)) if times else math.nan,
            "decision_ms_p95": float(np.percentile(times, 95)) if times else math.nan,
        })
    return rows


def nanmean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else math.nan


def summarize() -> dict:
    spec = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_frozen_files(spec)
    rows, ledger = [], []
    for lineup in LINEUPS:
        for variant in VARIANTS:
            for seed in SEEDS:
                raw_path, trace_path, manifest_path = paths(lineup, variant, seed)
                if not all(path.exists() for path in (raw_path, trace_path, manifest_path)):
                    raise ValueError(f"Missing run: {lineup}/{variant}/{seed}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest["protocol_sha256"] != digest(PROTOCOL_PATH):
                    raise ValueError(f"Protocol mismatch: {manifest_path}")
                if (
                    manifest["raw_sha256"] != digest(raw_path)
                    or manifest["trace_sha256"] != digest(trace_path)
                ):
                    raise ValueError(f"Result hash mismatch: {manifest_path}")
                ledger.append({
                    "lineup": lineup, "variant": variant, "seed": seed,
                    "manifest": str(manifest_path.relative_to(ROOT)),
                    "raw_sha256": digest(raw_path), "trace_sha256": digest(trace_path),
                })
                rows.extend(round_rows(lineup, variant, seed))

    identity = {"variant", "lineup", "seed", "round"}
    metric_names = [key for key in rows[0] if key not in identity]
    seed_rows = []
    for lineup in LINEUPS:
        for variant in VARIANTS:
            for seed in SEEDS:
                group = [
                    row for row in rows
                    if row["lineup"] == lineup
                    and row["variant"] == variant
                    and row["seed"] == seed
                ]
                seed_rows.append({
                    "lineup": lineup, "variant": variant, "seed": seed,
                    **{
                        metric: nanmean([row[metric] for row in group])
                        for metric in metric_names
                    },
                })

    aggregate = []
    for lineup in LINEUPS:
        for variant in VARIANTS:
            group = [
                row for row in rows
                if row["lineup"] == lineup and row["variant"] == variant
            ]
            all_times = []
            for seed in SEEDS:
                trace = json.loads(paths(lineup, variant, seed)[1].read_text())
                all_times.extend(
                    float(row["decision_ms"]) for row in trace["decisions"]
                    if row["decision_ms"] is not None
                )
            decisions = sum(row["decision_steps"] for row in group)
            opportunities = sum(row["opponent_route_opportunities"] for row in group)
            enabled = sum(row["opponent_route_enabled_steps"] for row in group)
            selections = sum(row["opponent_target_selections"] for row in group)
            selected_bombs = sum(row["selected_bombs"] for row in group)
            pressure_bombs = sum(row["trap_pressure_bombs"] for row in group)
            coin_routes = sum(row["coin_routing_activations"] for row in group)
            with_coin = sum(row["opponent_routes_with_coin"] for row in group)
            aggregate.append({
                "lineup": lineup, "variant": variant,
                "seed_clusters": len(SEEDS), "rounds": len(group),
                **{
                    metric: nanmean([row[metric] for row in group])
                    for metric in metric_names if metric != "decision_ms_p95"
                },
                "opponent_route_opportunity_rate": opportunities / max(decisions, 1),
                "opponent_route_enabled_rate": enabled / max(decisions, 1),
                "opponent_target_selection_rate": selections / max(opportunities, 1),
                "coin_routing_activation_rate": coin_routes / max(decisions, 1),
                "opponent_with_coin_rate": with_coin / max(opportunities, 1),
                "nonzero_trap_pressure_bomb_rate": pressure_bombs / max(selected_bombs, 1),
                "decision_ms_p95": float(np.percentile(all_times, 95)),
            })

    contrasts = (
        ("fallback_final", "opponent_route_off"),
        ("strict_hunt_only", "fallback_final"),
        ("strict_hunt_only", "opponent_route_off"),
    )
    comparison_metrics = [
        "score", "coins", "crates", "bombs", "kills", "suicides", "invalid",
        "survived", "opponent_route_opportunities", "opponent_route_enabled_steps",
        "opponent_target_selections", "opponent_progress", "trap_pressure_bombs",
        "bomb_opponent_yield", "coin_routing_activations",
        "opponent_routes_with_coin", "hunt_steps", "hunt_round",
        "self_trapped_after_bomb", "decision_ms_mean",
    ]
    comparisons = []
    rng = np.random.default_rng(spec["bootstrap"]["rng_seed"])
    for lineup in LINEUPS:
        for left_name, right_name in contrasts:
            for metric in comparison_metrics:
                differences = []
                for seed in SEEDS:
                    left = next(
                        row for row in seed_rows
                        if row["lineup"] == lineup
                        and row["variant"] == left_name and row["seed"] == seed
                    )
                    right = next(
                        row for row in seed_rows
                        if row["lineup"] == lineup
                        and row["variant"] == right_name and row["seed"] == seed
                    )
                    differences.append(left[metric] - right[metric])
                observed = np.asarray(differences, dtype=float)
                observed = observed[~np.isnan(observed)]
                if not len(observed):
                    comparisons.append({
                        "lineup": lineup, "contrast": f"{left_name}-{right_name}",
                        "metric": metric, "mean_difference": math.nan,
                        "ci_low": math.nan, "ci_high": math.nan,
                        "positive_seeds": 0, "zero_seeds": 0,
                        "negative_seeds": 0, "paired_seeds": 0,
                    })
                    continue
                indices = rng.integers(
                    0, len(observed),
                    size=(spec["bootstrap"]["replicates"], len(observed)),
                )
                samples = observed[indices].mean(axis=1)
                comparisons.append({
                    "lineup": lineup, "contrast": f"{left_name}-{right_name}",
                    "metric": metric, "mean_difference": float(np.mean(observed)),
                    "ci_low": float(np.percentile(samples, 2.5)),
                    "ci_high": float(np.percentile(samples, 97.5)),
                    "positive_seeds": int(np.sum(observed > 0)),
                    "zero_seeds": int(np.sum(observed == 0)),
                    "negative_seeds": int(np.sum(observed < 0)),
                    "paired_seeds": int(len(observed)),
                })

    payload = {
        "protocol": spec, "ledger": ledger, "round_rows": rows,
        "seed_rows": seed_rows, "aggregate": aggregate,
        "paired_bootstrap": comparisons,
    }
    atomic_json(SUMMARY_PATH, payload)
    return payload


def status() -> dict:
    complete, partial, missing = 0, [], []
    for lineup in LINEUPS:
        for variant in VARIANTS:
            for seed in SEEDS:
                run_paths = paths(lineup, variant, seed)
                count = sum(path.exists() for path in run_paths)
                if count == len(run_paths):
                    complete += 1
                elif count:
                    partial.append({"lineup": lineup, "variant": variant, "seed": seed})
                else:
                    missing.append({"lineup": lineup, "variant": variant, "seed": seed})
    return {
        "complete_clusters": complete,
        "total_clusters": len(LINEUPS) * len(VARIANTS) * len(SEEDS),
        "partial": partial, "missing": missing,
    }


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
        spec = protocol()
        if (
            PROTOCOL_PATH.exists()
            and json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")) != spec
        ):
            raise ValueError("E4 protocol already locked and differs")
        atomic_json(PROTOCOL_PATH, spec)
    elif args.command == "run":
        if args.seed not in SEEDS:
            raise ValueError("Seed outside locked E4 set")
        run_one(args.variant, args.lineup, args.seed)
    elif args.command == "summarize":
        summarize()
    else:
        print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    main_cli()
