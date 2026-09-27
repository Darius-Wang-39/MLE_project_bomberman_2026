"""Run the fresh-seed confirmatory E1 centre phase-gating experiment.

Three inference-only variants share one frozen policy and differ only in the
explicit centre Q term.  Five rounds are paired inside every lineup/seed block;
the seed-cluster average is the unit used by the paired bootstrap.
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

from evaluation.e3_economic_routing import classify_objective_source


CANDIDATE = "dueling_hunt_center_04"
VARIANTS = ("center_off", "center_hunt_04", "center_ungated_04")
LINEUPS = {
    "external_mixed": [
        "external_feature_is_everything",
        "external_deep_learning_killer",
        "rule_based_agent",
    ],
    "frozen_dddqn": [
        "dueling_ddqn_agent", "frozen_ddqn_day7", "frozen_ddqn_day11",
    ],
    "rule3": ["rule_based_agent"] * 3,
    "internal_mixed": [
        "frozen_ddqn_day7", "frozen_sarsa_best", "frozen_linear_q_final",
    ],
}
SEEDS = tuple(range(69001, 69041))
ROUNDS_PER_SEED = 5
RESULT_ROOT = ROOT / "results" / "e1_confirmatory_phase_gating"
PROTOCOL_PATH = RESULT_ROOT / "protocol.json"
SUMMARY_PATH = RESULT_ROOT / "summary.json"
ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
MOVE_DELTAS = {
    "UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1),
    "LEFT": (-1, 0), "WAIT": (0, 0), "BOMB": (0, 0),
}
ECONOMIC_SOURCES = {"coin", "immediate_crate_bomb", "productive_crate_target"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def center_distance(position) -> int:
    x, y = map(int, position)
    return max(3 - x, 0, x - 13) + max(3 - y, 0, y - 13)


def hunt_phase(game_state: dict) -> bool:
    return bool(
        not game_state["coins"]
        and not np.any(game_state["field"] == 1)
        and game_state["others"]
    )


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
        "experiment": "E1 confirmatory phase-gating of a weak centre prior",
        "question": (
            "Does restricting the same lambda=.04 centre prior to hunt phase "
            "improve outcomes relative to ungated use, and does hunt-only .04 "
            "have an observable effect relative to no centre prior?"
        ),
        "candidate": CANDIDATE,
        "checkpoint_sha256": digest(
            ROOT / "agent_code" / CANDIDATE / "dueling_ddqn_model.pt"
        ),
        "scenario": "classic",
        "variants": {
            "center_off": "lambda=0",
            "center_hunt_04": "lambda=.04; enabled only in hunt phase",
            "center_ungated_04": (
                "lambda=.04; retains the hunt-phase term and additionally enables "
                "it in every non-emergency non-hunt state, including states with "
                "coins or crates remaining"
            ),
        },
        "lineups": LINEUPS,
        "seeds": list(SEEDS),
        "seed_status": (
            "fresh confirmatory set; 69001-69040 are absent from E1-E4 and the "
            "repository seed ledger at protocol lock"
        ),
        "rounds_per_seed": ROUNDS_PER_SEED,
        "total_rounds": len(LINEUPS) * len(VARIANTS) * len(SEEDS) * ROUNDS_PER_SEED,
        "training": False,
        "exploration": 0.0,
        "pairing_unit": "lineup/seed cluster; average five rounds before inference",
        "prespecified_contrasts": [
            "center_hunt_04-center_ungated_04",
            "center_hunt_04-center_off",
        ],
        "primary_contrast": "center_hunt_04-center_ungated_04",
        "secondary_contrast": "center_hunt_04-center_off",
        "primary_metric": "mean_score",
        "explanatory_metrics": [
            "coins", "crates", "kills", "suicides", "invalid", "survived",
        ],
        "mechanism_metrics": [
            "center_activation_rate", "center_changed_action_rate",
            "economic_phase_center_interventions",
            "economic_route_overrides", "intervention_invalid_actions",
            "intervention_precenter_q_margin", "intervention_network_q_margin",
            "hunt_center_distance", "hunt_center_occupancy",
        ],
        "trace_definitions": {
            "center_activation": (
                "lambda is nonzero, the phase gate is open, and at least one safe "
                "action has nonzero centre progress"
            ),
            "center_changed_action": (
                "the selected action differs from a paired counterfactual selection "
                "using identical Q values and RNG state but zero centre term"
            ),
            "economic_phase_center_intervention": (
                "center_changed_action while at least one coin or crate remains"
            ),
            "economic_route_override": (
                "an economic intervention changes from an objective-target action "
                "to an action outside the selected coin/crate route"
            ),
            "raw_q_margin": (
                "chosen-centre action minus zero-centre action; both network-only "
                "and pre-centre adjusted-Q margins are logged"
            ),
        },
        "interpretation_rule": (
            "Treat gating as a conservative priority rule unless the paired score "
            "effect is directionally clear and stable across opponent pools. "
            "Mechanism effects do not by themselves establish performance gains."
        ),
        "bootstrap": {
            "unit": "paired seed cluster", "replicates": 10000, "rng_seed": 69099,
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


def explicit_center_progress(extractor, game_state: dict) -> np.ndarray:
    """Centre progress for every action, independent of hunt feature gating."""
    position = tuple(game_state["self"][-1])
    before = extractor.center_box_distance(position)
    legal = extractor.legal_action_mask(game_state)
    result = np.zeros(len(ACTIONS), dtype=np.float32)
    for index, action in enumerate(ACTIONS):
        if not legal[index]:
            continue
        dx, dy = MOVE_DELTAS[action]
        target = (position[0] + dx, position[1] + dy)
        result[index] = float(before - extractor.center_box_distance(target))
    return result


def make_act(callbacks, torch, variant: str, trace_rows: list):
    """Reproduce the frozen callback and log the zero-centre counterfactual."""
    lam = 0.0 if variant == "center_off" else 0.04

    def act(self, game_state: dict) -> str:
        allowed = self.feature_extractor.safe_action_mask(game_state)
        allowed_indices = np.flatnonzero(allowed)
        if allowed_indices.size == 0:
            return "WAIT"
        features = self.feature_extractor.action_features(game_state)
        source = classify_objective_source(self.feature_extractor, game_state, features)
        objective_index = callbacks.FEATURE_NAMES.index("objective_target")
        center_index = callbacks.FEATURE_NAMES.index("hunt_center_progress")
        objective = features[:, objective_index] > 0.5
        position = tuple(game_state["self"][-1])
        if self.position_round != game_state["round"]:
            self.recent_positions.clear()
            self.position_round = game_state["round"]

        revisit = np.zeros(len(ACTIONS), dtype=np.float32)
        danger_index = callbacks.FEATURE_NAMES.index("target_danger")
        is_hunt = hunt_phase(game_state)
        if is_hunt or not allowed[ACTIONS.index("BOMB")]:
            for index, action in enumerate(ACTIONS):
                if action == "BOMB" or features[index, danger_index] > 0.0:
                    continue
                dx, dy = callbacks.MOVE_DELTAS[action]
                target = (position[0] + dx, position[1] + dy)
                revisit[index] = callbacks.REVISIT_Q_PENALTY * min(
                    self.recent_positions.count(target), 3
                )
        self.recent_positions.append(position)

        state = torch.from_numpy(features.astype(np.float32).reshape(-1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            network_q = self.policy_net(state).squeeze(0).cpu().numpy()
        danger = float(self.feature_extractor._danger_score(game_state, position))
        # The submitted callback applies its hunt feature throughout hunt phase,
        # including danger states.  Ungated adds only non-hunt, non-emergency use.
        gate_open = bool(
            lam > 0.0
            and (
                is_hunt
                or (variant == "center_ungated_04" and danger == 0.0)
            )
        )
        if gate_open and variant == "center_ungated_04" and not is_hunt:
            center = explicit_center_progress(self.feature_extractor, game_state)
        elif gate_open:
            center = features[:, center_index].astype(np.float32).copy()
        else:
            center = np.zeros(len(ACTIONS), dtype=np.float32)

        precenter_q = (
            network_q + callbacks.OBJECTIVE_Q_BIAS * features[:, objective_index] - revisit
        )
        final_q = precenter_q + lam * center
        base_masked = np.where(allowed, precenter_q, -np.inf)
        final_masked = np.where(allowed, final_q, -np.inf)
        base_best = np.flatnonzero(np.isclose(base_masked, np.max(base_masked)))
        final_best = np.flatnonzero(np.isclose(final_masked, np.max(final_masked)))
        rng_state = self.rng.getstate()
        base_index = int(self.rng.choice(base_best.tolist()))
        self.rng.setstate(rng_state)
        chosen_index = int(self.rng.choice(final_best.tolist()))
        base_action, chosen = ACTIONS[base_index], ACTIONS[chosen_index]
        changed = chosen_index != base_index
        economic = bool(game_state["coins"]) or bool(np.any(game_state["field"] == 1))
        intervention = bool(changed and economic)
        route_override = bool(
            intervention and source in ECONOMIC_SOURCES
            and objective[base_index] and not objective[chosen_index]
        )
        before = center_distance(position)
        dx, dy = callbacks.MOVE_DELTAS[chosen]
        target = (position[0] + dx, position[1] + dy)
        trace_rows.append({
            "round": int(game_state["round"]), "step": int(game_state["step"]),
            "variant": variant, "action": chosen, "zero_center_action": base_action,
            "center_gate_open": gate_open,
            "center_activation": bool(gate_open and np.any(center[allowed] != 0.0)),
            "center_changed_action": changed,
            "economic_phase": economic, "economic_phase_center_intervention": intervention,
            "objective_source": source,
            "zero_center_action_on_route": bool(objective[base_index]),
            "chosen_action_on_route": bool(objective[chosen_index]),
            "economic_route_override": route_override,
            "network_q_margin": float(network_q[chosen_index] - network_q[base_index]),
            "precenter_q_margin": float(precenter_q[chosen_index] - precenter_q[base_index]),
            "center_bias_delta": float(lam * (center[chosen_index] - center[base_index])),
            "chosen_static_legal": bool(self.feature_extractor.legal_action_mask(game_state)[chosen_index]),
            "chosen_safe": bool(allowed[chosen_index]), "invalid_action": False,
            "hunt_phase": is_hunt, "center_distance": before,
            "inside_center_box": before == 0,
            "center_progress": float(before - center_distance(target)),
            "decision_ms": None,
        })
        return chosen

    return act


def run_one(variant: str, lineup: str, seed: int) -> None:
    import torch
    import events as e
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
    callbacks.act = make_act(callbacks, torch, variant, decisions)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    original_setup = BombeRLeWorld.setup_agents
    original_new = BombeRLeWorld.new_round
    original_wait = Agent.wait_for_act
    original_action = BombeRLeWorld.perform_agent_action
    current = {"candidate": None, "initial": []}

    def setup_agents(world, agents):
        original_setup(world, agents)
        candidate = world.agents[0]
        candidate.name = "candidate"
        candidate.display_name = "candidate"
        current["candidate"] = candidate

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
        result = original_action(world, agent, action)
        if agent is current["candidate"] and decisions:
            decisions[-1]["invalid_action"] = bool(e.INVALID_ACTION in agent.events)
        return result

    BombeRLeWorld.setup_agents = setup_agents
    BombeRLeWorld.new_round = new_round
    BombeRLeWorld.perform_agent_action = perform_agent_action
    Agent.wait_for_act = wait_for_act
    command = [
        "main.py", "play", "--no-gui", "--agents", CANDIDATE, *LINEUPS[lineup],
        "--scenario", "classic", "--n-rounds", str(ROUNDS_PER_SEED),
        "--seed", str(seed), "--save-stats", str(raw_path),
        "--match-name", f"e1confirm_{lineup}_{variant}_{seed}",
    ]
    main.main(command[1:])
    atomic_json(trace_path, {"decisions": decisions})
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
    trace = json.loads(trace_path.read_text(encoding="utf-8"))["decisions"]
    decisions = defaultdict(list)
    for row in trace:
        decisions[int(row["round"])].append(row)
    rows = []
    for round_id, round_data in enumerate(stats["by_round"].values(), start=1):
        outcome = round_data["agents"]["candidate"]
        rd = decisions[round_id]
        hunt = [row for row in rd if row["hunt_phase"]]
        interventions = [row for row in rd if row["economic_phase_center_intervention"]]
        route_overrides = [row for row in rd if row["economic_route_override"]]
        intervention_invalid = [row for row in interventions if row["invalid_action"]]
        changed = [row for row in rd if row["center_changed_action"]]
        times = [float(row["decision_ms"]) for row in rd if row["decision_ms"] is not None]

        def mean(group, key):
            return float(np.mean([row[key] for row in group])) if group else math.nan

        rows.append({
            "variant": variant, "lineup": lineup, "seed": seed, "round": round_id,
            "score": float(outcome["score"]), "coins": float(outcome["coins"]),
            "crates": float(outcome["crates"]), "bombs": float(outcome["bombs"]),
            "kills": float(outcome["kills"]), "suicides": float(outcome["suicides"]),
            "invalid": float(outcome["invalid"]), "survived": float(bool(outcome["survived"])),
            "decision_steps": float(len(rd)),
            "center_activations": float(sum(row["center_activation"] for row in rd)),
            "center_changed_actions": float(len(changed)),
            "economic_phase_center_interventions": float(len(interventions)),
            "economic_route_overrides": float(len(route_overrides)),
            "intervention_invalid_actions": float(len(intervention_invalid)),
            "intervention_precenter_q_margin": mean(interventions, "precenter_q_margin"),
            "intervention_network_q_margin": mean(interventions, "network_q_margin"),
            "intervention_center_bias_delta": mean(interventions, "center_bias_delta"),
            "hunt_steps": float(len(hunt)),
            "hunt_center_distance": mean(hunt, "center_distance"),
            "hunt_center_occupancy": mean(hunt, "inside_center_box"),
            "hunt_center_progress": mean(hunt, "center_progress"),
            "decision_ms_mean": float(np.mean(times)) if times else math.nan,
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

    identity = {"variant", "lineup", "seed", "round"}
    metric_names = [key for key in rows[0] if key not in identity]
    seed_rows = []
    for lineup in LINEUPS:
        for variant in VARIANTS:
            for seed in SEEDS:
                group = [
                    row for row in rows
                    if row["lineup"] == lineup and row["variant"] == variant and row["seed"] == seed
                ]
                seed_rows.append({
                    "lineup": lineup, "variant": variant, "seed": seed,
                    **{metric: nanmean([row[metric] for row in group]) for metric in metric_names},
                })

    aggregate = []
    for lineup in LINEUPS:
        for variant in VARIANTS:
            group = [row for row in rows if row["lineup"] == lineup and row["variant"] == variant]
            decisions = sum(row["decision_steps"] for row in group)
            interventions = sum(row["economic_phase_center_interventions"] for row in group)
            aggregate.append({
                "lineup": lineup, "variant": variant, "seed_clusters": len(SEEDS),
                "rounds": len(group),
                **{metric: nanmean([row[metric] for row in group]) for metric in metric_names},
                "center_activation_rate": sum(row["center_activations"] for row in group) / max(decisions, 1),
                "center_changed_action_rate": sum(row["center_changed_actions"] for row in group) / max(decisions, 1),
                "economic_route_override_rate": sum(row["economic_route_overrides"] for row in group) / max(interventions, 1),
                "intervention_invalid_rate": sum(row["intervention_invalid_actions"] for row in group) / max(interventions, 1),
            })

    comparison_metrics = [
        "score", "coins", "crates", "kills", "suicides", "invalid", "survived",
        "center_activations", "center_changed_actions",
        "economic_phase_center_interventions", "economic_route_overrides",
        "intervention_invalid_actions", "intervention_precenter_q_margin",
        "intervention_network_q_margin", "intervention_center_bias_delta",
        "hunt_center_distance", "hunt_center_occupancy", "hunt_center_progress",
    ]
    comparisons = []
    rng = np.random.default_rng(spec["bootstrap"]["rng_seed"])
    contrasts = (("center_hunt_04", "center_ungated_04"), ("center_hunt_04", "center_off"))
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
                if not len(observed):
                    low = high = estimate = math.nan
                else:
                    indices = rng.integers(0, len(observed), size=(spec["bootstrap"]["replicates"], len(observed)))
                    samples = observed[indices].mean(axis=1)
                    estimate = float(np.mean(observed))
                    low, high = map(float, np.percentile(samples, [2.5, 97.5]))
                comparisons.append({
                    "lineup": lineup, "contrast": f"{left_name}-{right_name}", "metric": metric,
                    "mean_difference": estimate, "ci_low": low, "ci_high": high,
                    "positive_seeds": int(np.sum(observed > 0)), "zero_seeds": int(np.sum(observed == 0)),
                    "negative_seeds": int(np.sum(observed < 0)), "paired_seeds": int(len(observed)),
                })
    payload = {
        "protocol": spec, "ledger": ledger, "round_rows": rows,
        "seed_rows": seed_rows, "aggregate": aggregate, "paired_bootstrap": comparisons,
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
            raise ValueError("Confirmatory E1 protocol already locked and differs")
        atomic_json(PROTOCOL_PATH, spec)
    elif args.command == "run":
        if args.seed not in SEEDS:
            raise ValueError("Seed outside locked confirmatory E1 set")
        run_one(args.variant, args.lineup, args.seed)
    elif args.command == "summarize":
        summarize()
    else:
        print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    main_cli()
