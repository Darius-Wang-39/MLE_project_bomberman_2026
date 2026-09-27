"""Run E3: phase-prioritised economic-routing ablation.

The frozen ``dueling_hunt_center_04`` network and safety layer are shared by all
arms.  Only the ``objective_target`` feature and its +1.0 direct Q bonus are
removed for selected economic objective sources.
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

CANDIDATE = "dueling_hunt_center_04"
VARIANTS = ("no_economic_route", "coin_only", "full_economic_route")
ECONOMIC_SOURCES = {
    "coin", "immediate_crate_bomb", "productive_crate_target",
}
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
SEEDS = tuple(range(67001, 67041))
ROUNDS_PER_SEED = 5
RESULT_ROOT = ROOT / "results" / "e3_economic_routing"
PROTOCOL_PATH = RESULT_ROOT / "protocol.json"
SUMMARY_PATH = RESULT_ROOT / "summary.json"
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


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
        "experiment": "E3 phase-prioritised economic-routing ablation",
        "question": (
            "Under the same multi-step safety layer, does coin-crate economic "
            "routing improve resource conversion, and at what offensive or safety cost?"
        ),
        "candidate": CANDIDATE,
        "checkpoint_sha256": digest(
            ROOT / "agent_code" / CANDIDATE / "dueling_ddqn_model.pt"
        ),
        "scenario": "classic",
        "variants": {
            "no_economic_route": (
                "clear objective_target for coin, immediate_crate_bomb, and "
                "productive_crate_target; retain escape and opponent routing"
            ),
            "coin_only": (
                "retain coin, escape, and opponent objective_target; clear both "
                "crate-related objective sources"
            ),
            "full_economic_route": "retain the frozen policy's complete routing",
        },
        "lineups": LINEUPS,
        "seeds": list(SEEDS),
        "seed_status": "fresh; not used by E1, E2, or the E2 development extension",
        "rounds_per_seed": ROUNDS_PER_SEED,
        "total_rounds": len(VARIANTS) * len(LINEUPS) * len(SEEDS) * ROUNDS_PER_SEED,
        "training": False,
        "exploration": 0.0,
        "pairing_unit": "lineup/seed cluster; average five rounds before inference",
        "shared_layers": [
            "network weights and architecture", "safe_action_mask",
            "hybrid bomb escape rule", "escape objective routing",
            "opponent objective routing", "hunt-centre Q bias",
            "revisit penalty",
        ],
        "manipulation": (
            "source-selective zeroing of objective_target before both network "
            "inference and the +1.0 direct Q bonus"
        ),
        "objective_sources": [
            "escape", "coin", "immediate_crate_bomb",
            "productive_crate_target", "opponent", "none",
        ],
        "prespecified_contrasts": [
            "full_economic_route-no_economic_route",
            "full_economic_route-coin_only",
        ],
        "primary_metric": "mean_score",
        "mechanism_metrics": [
            "coins", "crates", "bombs", "safe_productive_bombs",
            "economic_activation_rate", "economic_target_selection_rate",
            "coin_distance_reduction", "crate_distance_reduction",
        ],
        "tradeoff_metrics": [
            "kills", "hunt_steps", "opponent_activation_rate", "suicides",
            "invalid", "survived", "self_trapped_after_bomb", "decision_ms_p95",
        ],
        "bootstrap": {"unit": "paired seed cluster", "replicates": 10000, "rng_seed": 67099},
        "implementation_sha256": digest(Path(__file__)),
        "engine_files": {
            name: digest(ROOT / name)
            for name in ("main.py", "environment.py", "agents.py", "settings.py", "events.py", "items.py")
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


def classify_objective_source(extractor, game_state: dict, features: np.ndarray) -> str:
    """Reconstruct the frozen extractor's ordered objective-selection source."""
    names = extractor.feature_names
    objective = features[:, names.index("objective_target")] > 0.5
    if not objective.any():
        return "none"
    safe = extractor.safe_action_mask(game_state)
    safe_indices = np.flatnonzero(safe)
    current = tuple(game_state["self"][-1])
    if extractor._danger_score(game_state, current) > 0.0 and safe_indices.size:
        return "escape"
    coin_score = features[:, names.index("coin_progress")]
    if game_state["coins"] and safe_indices.size and np.max(coin_score[safe_indices]) > 0.0:
        return "coin"
    bomb_index = ACTIONS.index("BOMB")
    crate_yield = features[bomb_index, names.index("bomb_crate_yield")]
    if safe[bomb_index] and crate_yield > 0.0:
        return "immediate_crate_bomb"
    if np.any(game_state["field"] == 1):
        return "productive_crate_target"
    if game_state["others"]:
        return "opponent"
    return "none"


def source_enabled(variant: str, source: str) -> bool:
    if source not in ECONOMIC_SOURCES:
        return True
    if variant == "full_economic_route":
        return True
    if variant == "coin_only":
        return source == "coin"
    return False


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
        action_features = original_features.copy()
        if not source_enabled(variant, source):
            action_features[:, objective_index] = 0.0

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
        economic_active = source in ECONOMIC_SOURCES and original_objective.any()
        trace_rows.append({
            "round": int(game_state["round"]),
            "step": int(game_state["step"]),
            "variant": variant,
            "action": chosen,
            "objective_source": source,
            "objective_enabled": bool(source_enabled(variant, source)),
            "economic_activation": bool(economic_active),
            "objective_target_selected": bool(original_objective[chosen_index]),
            "coin_progress": float(original_features[chosen_index, callbacks.FEATURE_NAMES.index("coin_progress")]),
            "crate_progress": float(original_features[chosen_index, callbacks.FEATURE_NAMES.index("crate_progress")]),
            "safe_productive_bomb": bool(original_features[chosen_index, callbacks.FEATURE_NAMES.index("safe_productive_bomb")] > 0.5),
            "hunt_phase": bool(hunt_phase),
            "decision_ms": None,
        })
        return chosen
    return act


def run_one(variant: str, lineup: str, seed: int) -> None:
    import torch
    import main
    from agents import Agent
    from environment import BombeRLeWorld

    spec = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_frozen_files(spec)
    raw_path, trace_path, manifest_path = paths(lineup, variant, seed)
    if any(path.exists() for path in (raw_path, trace_path, manifest_path)):
        raise FileExistsError(f"Existing or partial run: {lineup}/{variant}/{seed}")
    for path in (raw_path, trace_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = importlib.import_module(f"agent_code.{CANDIDATE}.callbacks")
    decisions: list[dict] = []
    bombs: list[dict] = []
    callbacks.act = make_act(callbacks, torch, variant, decisions)
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    original_setup = BombeRLeWorld.setup_agents
    original_new = BombeRLeWorld.new_round
    original_wait = Agent.wait_for_act
    original_action = BombeRLeWorld.perform_agent_action
    original_poll = BombeRLeWorld.poll_and_run_agents
    original_evaluate = BombeRLeWorld.evaluate_explosions
    original_end = BombeRLeWorld.end_round
    current = {"candidate": None, "extractor": None, "initial": [], "pending": None, "counter": 0}

    def setup_agents(world, agents):
        original_setup(world, agents)
        candidate = world.agents[0]
        candidate.name = "candidate"
        candidate.display_name = "candidate"
        current["candidate"] = candidate
        current["extractor"] = candidate.backend.runner.fake_self.feature_extractor

    def new_round(world):
        round_seed = seed * 100 + world.round + 1
        world.rng = np.random.default_rng(round_seed)
        random.seed(round_seed)
        np.random.seed(round_seed)
        for index, agent in enumerate(world.agents):
            runtime = agent.backend.runner.fake_self
            if hasattr(runtime, "rng"):
                runtime.rng.seed(round_seed + index)
        original_new(world)
        current["pending"] = None
        initial = {
            "arena": world.arena.tolist(),
            "coins": [list(map(int, coin.get_state())) for coin in world.coins],
            "positions": [[int(agent.x), int(agent.y)] for agent in world.agents],
        }
        current["initial"].append({
            "round": int(world.round), "round_seed": int(round_seed),
            "sha256": hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest(),
        })

    def wait_for_act(agent):
        action, think_time = original_wait(agent)
        if agent is current["candidate"] and decisions:
            decisions[-1]["decision_ms"] = 1000.0 * float(think_time)
        return action, think_time

    def perform_agent_action(world, agent, action):
        placed = agent is current["candidate"] and action == "BOMB" and agent.bombs_left
        result = original_action(world, agent, action)
        if placed:
            current["counter"] += 1
            decision = decisions[-1]
            record = {
                "bomb_id": f"{lineup}:{variant}:{seed}:{world.round}:{current['counter']}",
                "round": int(world.round), "placed_step": int(world.step),
                "origin": [int(agent.x), int(agent.y)],
                "objective_source": decision["objective_source"],
                "safe_productive_bomb": decision["safe_productive_bomb"],
                "self_trapped_after_bomb": False, "first_trapped_step": None,
                "death_cause": None, "death_step": None,
            }
            bombs.append(record)
            current["pending"] = record
        return result

    def poll_and_run_agents(world):
        result = original_poll(world)
        candidate = current["candidate"]
        pending = current["pending"]
        if pending is not None and candidate in world.active_agents:
            own_bomb_present = any(
                bomb.owner is candidate and [int(bomb.x), int(bomb.y)] == pending["origin"]
                for bomb in world.bombs
            )
            if own_bomb_present:
                state = world.get_state_for_agent(candidate)
                extractor = current["extractor"]
                legal = extractor.legal_action_mask(state)
                viable = [
                    action for index, action in enumerate(ACTIONS)
                    if legal[index] and extractor.survival_count(state, action) > 0
                ]
                if not viable and not pending["self_trapped_after_bomb"]:
                    pending["self_trapped_after_bomb"] = True
                    pending["first_trapped_step"] = int(world.step)
            else:
                current["pending"] = None
        return result

    def evaluate_explosions(world):
        candidate = current["candidate"]
        was_alive = candidate in world.active_agents
        owners = []
        if was_alive:
            for explosion in world.explosions:
                if explosion.is_dangerous() and (candidate.x, candidate.y) in explosion.blast_coords:
                    owners.append(explosion.owner)
        result = original_evaluate(world)
        pending = current["pending"]
        if was_alive and candidate not in world.active_agents and pending is not None:
            own = any(owner is candidate for owner in owners)
            other = any(owner is not candidate for owner in owners)
            pending["death_cause"] = (
                "own_and_opponent_bomb" if own and other else
                "own_bomb_suicide" if own else
                "opponent_bomb" if other else "other"
            )
            pending["death_step"] = int(world.step)
        return result

    def end_round(world):
        current["pending"] = None
        return original_end(world)

    BombeRLeWorld.setup_agents = setup_agents
    BombeRLeWorld.new_round = new_round
    Agent.wait_for_act = wait_for_act
    BombeRLeWorld.perform_agent_action = perform_agent_action
    BombeRLeWorld.poll_and_run_agents = poll_and_run_agents
    BombeRLeWorld.evaluate_explosions = evaluate_explosions
    BombeRLeWorld.end_round = end_round
    command = [
        "main.py", "play", "--no-gui", "--agents", CANDIDATE, *LINEUPS[lineup],
        "--scenario", "classic", "--n-rounds", str(ROUNDS_PER_SEED),
        "--seed", str(seed), "--save-stats", str(raw_path),
        "--match-name", f"e3_{lineup}_{variant}_{seed}",
    ]
    main.main(command[1:])
    atomic_json(trace_path, {"trace_version": 1, "decisions": decisions, "bombs": bombs})
    atomic_json(manifest_path, {
        "variant": variant, "lineup": lineup, "seed": seed,
        "rounds": ROUNDS_PER_SEED, "training": False, "command": command,
        "initial_conditions": current["initial"],
        "raw_stats": str(raw_path.relative_to(ROOT)), "raw_sha256": digest(raw_path),
        "trace": str(trace_path.relative_to(ROOT)), "trace_sha256": digest(trace_path),
        "protocol_sha256": digest(PROTOCOL_PATH),
    })
    validate_frozen_files(spec)


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
        economic = [row for row in rd if row["economic_activation"]]
        coin = [row for row in economic if row["objective_source"] == "coin"]
        crate = [row for row in economic if row["objective_source"] in {"immediate_crate_bomb", "productive_crate_target"}]
        opponent = [row for row in rd if row["objective_source"] == "opponent"]
        times = [float(row["decision_ms"]) for row in rd if row["decision_ms"] is not None]
        rows.append({
            "variant": variant, "lineup": lineup, "seed": seed, "round": round_id,
            "score": float(outcome["score"]), "coins": float(outcome["coins"]),
            "crates": float(outcome["crates"]), "bombs": float(outcome["bombs"]),
            "kills": float(outcome["kills"]), "suicides": float(outcome["suicides"]),
            "invalid": float(outcome["invalid"]), "survived": float(bool(outcome["survived"])),
            "safe_productive_bombs": float(sum(row["safe_productive_bomb"] for row in rb)),
            "economic_activations": float(len(economic)),
            "economic_target_selections": float(sum(row["objective_target_selected"] for row in economic)),
            "decision_steps": float(len(rd)), "hunt_steps": float(sum(row["hunt_phase"] for row in rd)),
            "opponent_activations": float(len(opponent)),
            "coin_distance_reduction": float(np.mean([row["coin_progress"] for row in coin])) if coin else math.nan,
            "crate_distance_reduction": float(np.mean([row["crate_progress"] for row in crate])) if crate else math.nan,
            "self_trapped_after_bomb": float(sum(row["self_trapped_after_bomb"] for row in rb)),
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
                if manifest["raw_sha256"] != digest(raw_path) or manifest["trace_sha256"] != digest(trace_path):
                    raise ValueError(f"Result hash mismatch: {manifest_path}")
                ledger.append({
                    "lineup": lineup, "variant": variant, "seed": seed,
                    "manifest": str(manifest_path.relative_to(ROOT)),
                    "raw_sha256": digest(raw_path), "trace_sha256": digest(trace_path),
                })
                rows.extend(round_rows(lineup, variant, seed))

    metric_names = [key for key in rows[0] if key not in {"variant", "lineup", "seed", "round"}]
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
            group = [row for row in rows if row["lineup"] == lineup and row["variant"] == variant]
            all_times = []
            for seed in SEEDS:
                trace = json.loads(paths(lineup, variant, seed)[1].read_text(encoding="utf-8"))
                all_times.extend(float(row["decision_ms"]) for row in trace["decisions"] if row["decision_ms"] is not None)
            activations = sum(row["economic_activations"] for row in group)
            decisions = sum(row["decision_steps"] for row in group)
            opponent = sum(row["opponent_activations"] for row in group)
            target_selections = sum(row["economic_target_selections"] for row in group)
            aggregate.append({
                "lineup": lineup, "variant": variant, "seed_clusters": len(SEEDS), "rounds": len(group),
                **{metric: nanmean([row[metric] for row in group]) for metric in metric_names if metric != "decision_ms_p95"},
                "economic_activation_rate": activations / max(decisions, 1),
                "economic_target_selection_rate": target_selections / max(activations, 1),
                "opponent_activation_rate": opponent / max(decisions, 1),
                "decision_ms_p95": float(np.percentile(all_times, 95)),
            })

    contrasts = (
        ("full_economic_route", "no_economic_route"),
        ("full_economic_route", "coin_only"),
    )
    comparison_metrics = [
        "score", "coins", "crates", "bombs", "safe_productive_bombs",
        "economic_target_selections", "coin_distance_reduction", "crate_distance_reduction",
        "kills", "hunt_steps", "opponent_activations", "suicides", "invalid",
        "survived", "self_trapped_after_bomb", "decision_ms_mean",
    ]
    comparisons = []
    rng = np.random.default_rng(spec["bootstrap"]["rng_seed"])
    for lineup in LINEUPS:
        for left_name, right_name in contrasts:
            for metric in comparison_metrics:
                differences = []
                for seed in SEEDS:
                    left = next(row for row in seed_rows if row["lineup"] == lineup and row["variant"] == left_name and row["seed"] == seed)
                    right = next(row for row in seed_rows if row["lineup"] == lineup and row["variant"] == right_name and row["seed"] == seed)
                    differences.append(left[metric] - right[metric])
                observed = np.asarray(differences, dtype=float)
                observed = observed[~np.isnan(observed)]
                indices = rng.integers(0, len(observed), size=(spec["bootstrap"]["replicates"], len(observed)))
                samples = observed[indices].mean(axis=1)
                comparisons.append({
                    "lineup": lineup, "contrast": f"{left_name}-{right_name}", "metric": metric,
                    "mean_difference": float(np.mean(observed)),
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
        if PROTOCOL_PATH.exists() and json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")) != spec:
            raise ValueError("E3 protocol already locked and differs")
        atomic_json(PROTOCOL_PATH, spec)
    elif args.command == "run":
        if args.seed not in SEEDS:
            raise ValueError("Seed outside locked E3 set")
        run_one(args.variant, args.lineup, args.seed)
    elif args.command == "summarize":
        summarize()
    else:
        print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    main_cli()
