"""Run E2: two versus one first-step bomb-escape branches.

The frozen ``dueling_hunt_center_04`` policy is evaluated with zero exploration.
The two arms differ only in the minimum number of independently survivable
first moves required before BOMB is included in the final action mask.
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
VARIANTS = {
    "two_branch_final": 2,
    "one_branch_relaxed": 1,
}
LINEUPS = {
    "external_mixed": [
        "external_feature_is_everything",
        "external_deep_learning_killer",
        "rule_based_agent",
    ],
    "frozen_dddqn": [
        "dueling_ddqn_agent",
        "frozen_ddqn_day7",
        "frozen_ddqn_day11",
    ],
    "internal_mixed": [
        "frozen_ddqn_day7",
        "frozen_sarsa_best",
        "frozen_linear_q_final",
    ],
}
CORE_LINEUPS = ("external_mixed", "frozen_dddqn")
ROBUSTNESS_LINEUPS = ("internal_mixed",)
SEEDS = tuple(range(66001, 66041))
ROUNDS_PER_SEED = 5
RESULT_ROOT = ROOT / "results" / "e2_bomb_escape_branches"
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


def frozen_files(lineups=CORE_LINEUPS) -> dict[str, str]:
    agents = {CANDIDATE}
    for lineup in lineups:
        agents.update(LINEUPS[lineup])
    paths = []
    for agent in sorted(agents):
        base = ROOT / "agent_code" / agent
        paths.extend(
            path
            for path in base.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and "logs" not in path.parts
            and path.suffix.lower() in {".py", ".pt", ".npz", ".json"}
        )
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def protocol(lineups=CORE_LINEUPS) -> dict:
    lineups = tuple(lineups)
    return {
        "experiment": "E2 active multi-step bomb-escape validation",
        "question": (
            "Does requiring two independent first-step bomb-escape branches reduce "
            "catastrophic self-traps enough to justify potentially fewer productive bombs?"
        ),
        "candidate": CANDIDATE,
        "checkpoint_sha256": digest(
            ROOT / "agent_code" / CANDIDATE / "dueling_ddqn_model.pt"
        ),
        "scenario": "classic",
        "variants": {
            name: {"minimum_independent_first_step_escape_branches": threshold}
            for name, threshold in VARIANTS.items()
        },
        "lineups": {name: LINEUPS[name] for name in lineups},
        "core_lineups": list(CORE_LINEUPS),
        "optional_robustness_lineups": list(ROBUSTNESS_LINEUPS),
        "seeds": list(SEEDS),
        "seed_status": "fresh; not used in E1/E1 phase-gating",
        "rounds_per_seed": ROUNDS_PER_SEED,
        "total_rounds": len(lineups) * len(VARIANTS) * len(SEEDS) * ROUNDS_PER_SEED,
        "training": False,
        "exploration": 0.0,
        "pairing_unit": (
            "lineup/seed cluster; the five rounds are averaged before paired inference"
        ),
        "only_manipulation": (
            "BOMB enters the final permission mask iff its independent first-step "
            "escape-branch count is at least 2 versus at least 1"
        ),
        "locked_inference": {
            "immediate_lethality_filter": "legal action with survival_count > 0",
            "objective_q_bias": 1.0,
            "hunt_center_q_bias": 0.04,
            "danger_model": "frozen candidate FeatureExtractor",
            "opponents": "fixed by lineup",
        },
        "trace_definitions": {
            "bomb_proposal": (
                "BOMB is among the maximum adjusted-Q actions after the common "
                "immediate-lethality filter but before the branch threshold"
            ),
            "bomb_rejected": "bomb_proposal and branch count below the arm threshold",
            "self_trapped_after_bomb": (
                "after an actually placed bomb and before it explodes, an observed "
                "post-action world state has no legal action with a survivable continuation"
            ),
        },
        "estimands": {
            "primary": "mean_score",
            "productivity": ["bombs", "crates", "coins"],
            "safety": ["self_trapped_after_bomb", "suicides", "survived"],
            "engineering": ["decision_ms_p95"],
            "exposure": ["bomb_proposals", "bomb_rejections", "actual_bombs"],
            "contrast": "two_branch_final - one_branch_relaxed",
        },
        "bootstrap": {"unit": "paired seed cluster", "replicates": 10000, "rng_seed": 66099},
        "implementation_sha256": digest(Path(__file__)),
        "engine_files": {
            name: digest(ROOT / name)
            for name in ("main.py", "environment.py", "agents.py", "settings.py", "events.py", "items.py")
        },
        "frozen_files": frozen_files(lineups),
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


def immediate_survival_mask(extractor, game_state: dict) -> tuple[np.ndarray, np.ndarray]:
    """The common passive filter, deliberately excluding branch redundancy."""
    legal = extractor.legal_action_mask(game_state)
    survival = np.asarray(
        [
            extractor.survival_count(game_state, action) if legal[index] else 0
            for index, action in enumerate(ACTIONS)
        ],
        dtype=int,
    )
    allowed = legal & (survival > 0)
    if not allowed.any():
        allowed = legal.copy()
        allowed[ACTIONS.index("BOMB")] = False
        if not allowed.any():
            allowed = legal
    return allowed, survival


def apply_branch_rule(
    allowed: np.ndarray, branch_count: int, minimum_branches: int
) -> np.ndarray:
    result = allowed.copy()
    bomb_index = ACTIONS.index("BOMB")
    if branch_count < minimum_branches:
        result[bomb_index] = False
    return result


def make_instrumented_act(callbacks, torch, minimum_branches: int, trace_rows: list):
    """Reproduce the frozen callback, changing only final BOMB permission."""

    def act(self, game_state: dict) -> str:
        proposal_allowed, survival = immediate_survival_mask(
            self.feature_extractor, game_state
        )
        branch_count = self.feature_extractor.bomb_escape_branch_count(game_state)
        allowed = apply_branch_rule(proposal_allowed, branch_count, minimum_branches)
        allowed_indices = np.flatnonzero(allowed)
        if allowed_indices.size == 0:
            allowed[ACTIONS.index("WAIT")] = True
            allowed_indices = np.flatnonzero(allowed)

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
            for index, candidate_action in enumerate(ACTIONS):
                if (
                    candidate_action == "BOMB"
                    or action_features[index, danger_index] > 0.0
                ):
                    continue
                target = (
                    position[0] + callbacks.MOVE_DELTAS[candidate_action][0],
                    position[1] + callbacks.MOVE_DELTAS[candidate_action][1],
                )
                revisit_penalties[index] = callbacks.REVISIT_Q_PENALTY * min(
                    self.recent_positions.count(target), 3
                )
        self.recent_positions.append(position)

        state = (
            torch.from_numpy(action_features.astype(np.float32).reshape(-1))
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze(0).cpu().numpy()
        center_index = callbacks.FEATURE_NAMES.index("hunt_center_progress")
        objective_index = callbacks.FEATURE_NAMES.index("objective_target")
        q_values = (
            q_values
            + self.hunt_center_q_bias * action_features[:, center_index]
            + callbacks.OBJECTIVE_Q_BIAS * action_features[:, objective_index]
            - revisit_penalties
        )

        proposal_q = np.where(proposal_allowed, q_values, -np.inf)
        proposal_best = np.flatnonzero(
            np.isclose(proposal_q, np.max(proposal_q))
        )
        bomb_index = ACTIONS.index("BOMB")
        bomb_proposal = bool(bomb_index in proposal_best)
        bomb_permitted = bool(allowed[bomb_index])

        masked_q = np.where(allowed, q_values, -np.inf)
        best = np.flatnonzero(np.isclose(masked_q, np.max(masked_q)))
        chosen = ACTIONS[self.rng.choice(best.tolist())]
        trace_rows.append(
            {
                "round": int(game_state["round"]),
                "step": int(game_state["step"]),
                "action": chosen,
                "bomb_proposal": bomb_proposal,
                "bomb_permitted": bomb_permitted,
                "bomb_rejected": bool(bomb_proposal and not bomb_permitted),
                "bomb_branch_count": int(branch_count),
                "bomb_survival_count": int(survival[bomb_index]),
                "decision_ms": None,
            }
        )
        return chosen

    return act


def run_one(variant: str, lineup: str, seed: int) -> None:
    import torch
    import main
    from agents import Agent
    from environment import BombeRLeWorld

    spec = json.loads((RESULT_ROOT / "protocol.json").read_text(encoding="utf-8"))
    if lineup not in spec["lineups"]:
        raise ValueError(f"Lineup {lineup} is not in the locked protocol")
    validate_frozen_files(spec)
    raw_path, trace_path, manifest_path = paths(lineup, variant, seed)
    if any(path.exists() for path in (raw_path, trace_path, manifest_path)):
        raise FileExistsError(f"Existing or partial run: {lineup}/{variant}/{seed}")
    for path in (raw_path, trace_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = importlib.import_module(f"agent_code.{CANDIDATE}.callbacks")
    decision_rows: list[dict] = []
    bomb_rows: list[dict] = []
    callbacks.act = make_instrumented_act(
        callbacks, torch, VARIANTS[variant], decision_rows
    )
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    original_setup = BombeRLeWorld.setup_agents
    original_new = BombeRLeWorld.new_round
    original_poll = BombeRLeWorld.poll_and_run_agents
    original_action = BombeRLeWorld.perform_agent_action
    original_wait = Agent.wait_for_act
    current = {
        "candidate": None,
        "extractor": None,
        "initial": [],
        "pending_bomb": None,
    }

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
        current["pending_bomb"] = None
        initial = {
            "arena": world.arena.tolist(),
            "coins": [list(map(int, coin.get_state())) for coin in world.coins],
            "positions": [[int(agent.x), int(agent.y)] for agent in world.agents],
        }
        current["initial"].append(
            {
                "round": int(world.round),
                "round_seed": int(round_seed),
                "sha256": hashlib.sha256(
                    json.dumps(initial, sort_keys=True).encode()
                ).hexdigest(),
            }
        )

    def wait_for_act(agent):
        action, think_time = original_wait(agent)
        if agent is current["candidate"] and decision_rows:
            decision_rows[-1]["decision_ms"] = 1000.0 * float(think_time)
        return action, think_time

    def perform_agent_action(world, agent, action):
        successfully_armed = (
            agent is current["candidate"] and action == "BOMB" and agent.bombs_left
        )
        result = original_action(world, agent, action)
        if successfully_armed:
            bomb = {
                "round": int(world.round),
                "placed_step": int(world.step),
                "origin": [int(agent.x), int(agent.y)],
                "self_trapped_after_bomb": False,
                "first_trapped_step": None,
            }
            bomb_rows.append(bomb)
            current["pending_bomb"] = bomb
        return result

    def poll_and_run_agents(world):
        result = original_poll(world)
        candidate = current["candidate"]
        pending = current["pending_bomb"]
        if pending is not None and candidate in world.active_agents:
            own_bomb_present = any(
                bomb.owner is candidate
                and [int(bomb.x), int(bomb.y)] == pending["origin"]
                for bomb in world.bombs
            )
            if own_bomb_present:
                state = world.get_state_for_agent(candidate)
                legal = current["extractor"].legal_action_mask(state)
                viable = any(
                    legal[index]
                    and current["extractor"].survival_count(state, action) > 0
                    for index, action in enumerate(ACTIONS)
                )
                if not viable and not pending["self_trapped_after_bomb"]:
                    pending["self_trapped_after_bomb"] = True
                    pending["first_trapped_step"] = int(world.step)
            else:
                current["pending_bomb"] = None
        return result

    BombeRLeWorld.setup_agents = setup_agents
    BombeRLeWorld.new_round = new_round
    BombeRLeWorld.poll_and_run_agents = poll_and_run_agents
    BombeRLeWorld.perform_agent_action = perform_agent_action
    Agent.wait_for_act = wait_for_act
    command = [
        "main.py",
        "play",
        "--no-gui",
        "--agents",
        CANDIDATE,
        *LINEUPS[lineup],
        "--scenario",
        "classic",
        "--n-rounds",
        str(ROUNDS_PER_SEED),
        "--seed",
        str(seed),
        "--save-stats",
        str(raw_path),
        "--match-name",
        f"e2_{lineup}_{variant}_{seed}",
    ]
    main.main(command[1:])
    trace_payload = {"decisions": decision_rows, "bombs": bomb_rows}
    atomic_json(trace_path, trace_payload)
    atomic_json(
        manifest_path,
        {
            "variant": variant,
            "minimum_branches": VARIANTS[variant],
            "lineup": lineup,
            "seed": seed,
            "rounds": ROUNDS_PER_SEED,
            "training": False,
            "command": command,
            "initial_conditions": current["initial"],
            "raw_stats": str(raw_path.relative_to(ROOT)),
            "raw_sha256": digest(raw_path),
            "trace": str(trace_path.relative_to(ROOT)),
            "trace_sha256": digest(trace_path),
            "protocol_sha256": digest(RESULT_ROOT / "protocol.json"),
        },
    )
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
        round_decisions = decisions[round_id]
        round_bombs = bombs[round_id]
        times = [
            float(row["decision_ms"])
            for row in round_decisions
            if row["decision_ms"] is not None
        ]
        rows.append(
            {
                "variant": variant,
                "lineup": lineup,
                "seed": seed,
                "round": round_id,
                "score": float(outcome["score"]),
                "coins": float(outcome["coins"]),
                "crates": float(outcome["crates"]),
                "bombs": float(outcome["bombs"]),
                "kills": float(outcome["kills"]),
                "suicides": float(outcome["suicides"]),
                "invalid": float(outcome["invalid"]),
                "survived": float(bool(outcome["survived"])),
                "bomb_proposals": float(sum(row["bomb_proposal"] for row in round_decisions)),
                "bomb_rejections": float(sum(row["bomb_rejected"] for row in round_decisions)),
                "actual_bombs": float(len(round_bombs)),
                "self_trapped_after_bomb": float(
                    sum(row["self_trapped_after_bomb"] for row in round_bombs)
                ),
                "decision_ms_mean": float(np.mean(times)) if times else math.nan,
                "decision_ms_p95": float(np.percentile(times, 95)) if times else math.nan,
            }
        )
    return rows


def nanmean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else math.nan


def summarize() -> dict:
    spec = json.loads((RESULT_ROOT / "protocol.json").read_text(encoding="utf-8"))
    validate_frozen_files(spec)
    lineups = tuple(spec["lineups"])
    rows = []
    ledger = []
    for lineup in lineups:
        for variant in VARIANTS:
            for seed in SEEDS:
                raw_path, trace_path, manifest_path = paths(lineup, variant, seed)
                if not all(path.exists() for path in (raw_path, trace_path, manifest_path)):
                    raise ValueError(f"Missing run: {lineup}/{variant}/{seed}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest["protocol_sha256"] != digest(RESULT_ROOT / "protocol.json"):
                    raise ValueError(f"Protocol mismatch: {manifest_path}")
                if manifest["raw_sha256"] != digest(raw_path) or manifest["trace_sha256"] != digest(trace_path):
                    raise ValueError(f"Result hash mismatch: {manifest_path}")
                ledger.append(
                    {
                        "lineup": lineup,
                        "variant": variant,
                        "seed": seed,
                        "manifest": str(manifest_path.relative_to(ROOT)),
                        "raw_sha256": digest(raw_path),
                        "trace_sha256": digest(trace_path),
                    }
                )
                rows.extend(round_rows(lineup, variant, seed))

    metric_names = [
        key
        for key in rows[0]
        if key not in {"variant", "lineup", "seed", "round"}
    ]
    seed_rows = []
    for lineup in lineups:
        for variant in VARIANTS:
            for seed in SEEDS:
                group = [
                    row
                    for row in rows
                    if row["lineup"] == lineup
                    and row["variant"] == variant
                    and row["seed"] == seed
                ]
                seed_rows.append(
                    {
                        "lineup": lineup,
                        "variant": variant,
                        "seed": seed,
                        **{
                            metric: nanmean([row[metric] for row in group])
                            for metric in metric_names
                        },
                    }
                )

    aggregate = []
    for lineup in lineups:
        for variant in VARIANTS:
            group = [
                row
                for row in rows
                if row["lineup"] == lineup and row["variant"] == variant
            ]
            all_times = []
            for seed in SEEDS:
                trace = json.loads(paths(lineup, variant, seed)[1].read_text(encoding="utf-8"))
                all_times.extend(
                    float(row["decision_ms"])
                    for row in trace["decisions"]
                    if row["decision_ms"] is not None
                )
            aggregate.append(
                {
                    "lineup": lineup,
                    "variant": variant,
                    "seed_clusters": len(SEEDS),
                    "rounds": len(group),
                    **{
                        metric: nanmean([row[metric] for row in group])
                        for metric in metric_names
                        if metric != "decision_ms_p95"
                    },
                    "decision_ms_p95": float(np.percentile(all_times, 95)),
                    "bomb_proposals_total": int(sum(row["bomb_proposals"] for row in group)),
                    "bomb_rejections_total": int(sum(row["bomb_rejections"] for row in group)),
                    "actual_bombs_total": int(sum(row["actual_bombs"] for row in group)),
                    "self_trapped_total": int(sum(row["self_trapped_after_bomb"] for row in group)),
                }
            )

    comparison_metrics = [
        "score",
        "coins",
        "crates",
        "bombs",
        "suicides",
        "survived",
        "bomb_proposals",
        "bomb_rejections",
        "actual_bombs",
        "self_trapped_after_bomb",
        "decision_ms_mean",
    ]
    comparisons = []
    rng = np.random.default_rng(spec["bootstrap"]["rng_seed"])
    for lineup in lineups:
        for metric in comparison_metrics:
            differences = []
            for seed in SEEDS:
                left = next(
                    row
                    for row in seed_rows
                    if row["lineup"] == lineup
                    and row["variant"] == "two_branch_final"
                    and row["seed"] == seed
                )
                right = next(
                    row
                    for row in seed_rows
                    if row["lineup"] == lineup
                    and row["variant"] == "one_branch_relaxed"
                    and row["seed"] == seed
                )
                differences.append(left[metric] - right[metric])
            observed = np.asarray(differences, dtype=float)
            observed = observed[~np.isnan(observed)]
            indices = rng.integers(
                0,
                len(observed),
                size=(spec["bootstrap"]["replicates"], len(observed)),
            )
            samples = observed[indices].mean(axis=1)
            comparisons.append(
                {
                    "lineup": lineup,
                    "contrast": "two_branch_final-one_branch_relaxed",
                    "metric": metric,
                    "mean_difference": float(np.mean(observed)),
                    "ci_low": float(np.percentile(samples, 2.5)),
                    "ci_high": float(np.percentile(samples, 97.5)),
                    "positive_seeds": int(np.sum(observed > 0)),
                    "zero_seeds": int(np.sum(observed == 0)),
                    "negative_seeds": int(np.sum(observed < 0)),
                    "paired_seeds": int(len(observed)),
                }
            )

    payload = {
        "protocol": spec,
        "ledger": ledger,
        "round_rows": rows,
        "seed_rows": seed_rows,
        "aggregate": aggregate,
        "paired_bootstrap": comparisons,
    }
    atomic_json(RESULT_ROOT / "summary.json", payload)
    return payload


def status() -> dict:
    protocol_path = RESULT_ROOT / "protocol.json"
    lineups = CORE_LINEUPS
    if protocol_path.exists():
        spec = json.loads(protocol_path.read_text(encoding="utf-8"))
        lineups = tuple(spec["lineups"])
    complete = 0
    partial = []
    missing = []
    for lineup in lineups:
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
        "total_clusters": len(lineups) * len(VARIANTS) * len(SEEDS),
        "partial": partial,
        "missing": missing,
    }


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument(
        "--include-robustness", action="store_true", help="also lock internal_mixed"
    )
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--variant", choices=VARIANTS, required=True)
    run_parser.add_argument("--lineup", choices=LINEUPS, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    sub.add_parser("summarize")
    sub.add_parser("status")
    args = parser.parse_args()

    if args.command == "prepare":
        lineups = CORE_LINEUPS + ROBUSTNESS_LINEUPS if args.include_robustness else CORE_LINEUPS
        target = RESULT_ROOT / "protocol.json"
        spec = protocol(lineups)
        if target.exists() and json.loads(target.read_text(encoding="utf-8")) != spec:
            raise ValueError("E2 protocol is already locked and differs from current inputs")
        atomic_json(target, spec)
    elif args.command == "run":
        if args.seed not in SEEDS:
            raise ValueError("Seed is outside the locked protocol")
        spec = json.loads((RESULT_ROOT / "protocol.json").read_text(encoding="utf-8"))
        if args.lineup not in spec["lineups"]:
            raise ValueError("Lineup is outside the locked protocol")
        run_one(args.variant, args.lineup, args.seed)
    elif args.command == "summarize":
        summarize()
    else:
        print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    main_cli()
