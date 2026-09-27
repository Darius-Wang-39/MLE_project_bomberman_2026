"""Development screen for contextual one-branch bomb-permission rules.

This is an explicitly non-confirmatory extension using already-viewed E2 seeds.
Four arms are rerun under one enhanced trace pipeline: relaxed, original hybrid,
phase/productive hybrid, and armed-threat hybrid.  Strict two-branch is archived
and deliberately absent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import e2_bomb_escape_branches as base_e2


CANDIDATE = "dueling_hunt_center_04"
VARIANTS = (
    "one_branch_relaxed",
    "original_hybrid",
    "phase_productive_hybrid",
    "armed_threat_hybrid",
)
CORE_LINEUP_NAMES = tuple(base_e2.CORE_LINEUPS)
ROBUSTNESS_LINEUP_NAMES = ("rule_based", "internal_mixed")
LINEUPS = {
    **{name: base_e2.LINEUPS[name] for name in CORE_LINEUP_NAMES},
    "rule_based": ["rule_based_agent", "rule_based_agent", "rule_based_agent"],
    "internal_mixed": list(base_e2.LINEUPS["internal_mixed"]),
}
SEEDS = base_e2.SEEDS
ROUNDS_PER_SEED = base_e2.ROUNDS_PER_SEED
RESULT_ROOT = ROOT / "results" / "e2_contextual_hybrid_screen"
PROTOCOL_PATH = RESULT_ROOT / "protocol.json"
ROBUSTNESS_PROTOCOL_PATH = RESULT_ROOT / "protocol_robustness.json"
SUMMARY_PATH = RESULT_ROOT / "summary.json"
STRICT_ATTRIBUTION_CSV = RESULT_ROOT / "strict_self_trap_attribution.csv"
STRICT_ROUND_ROWS_CSV = RESULT_ROOT / "strict_self_trap_round_rows.csv"
STRICT_BCI_CSV = RESULT_ROOT / "strict_self_trap_pairwise_bci.csv"
ACTIONS = base_e2.ACTIONS
MOVE_DELTAS = {
    "UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1),
    "LEFT": (-1, 0), "WAIT": (0, 0), "BOMB": (0, 0),
}
CARDINAL_ACTIONS = ACTIONS[:4]
NEAR_DISTANCE = 5
ORIGINAL_MAX_STEP = 120
CONTRASTS = tuple(
    (VARIANTS[left], VARIANTS[right])
    for left in range(len(VARIANTS))
    for right in range(left + 1, len(VARIANTS))
)
REJECTION_KEYS = (
    "reject_no_escape_branch",
    "reject_nonproductive",
    "reject_no_crates_remaining",
    "reject_step_cutoff",
    "reject_active_bomb",
    "reject_active_explosion",
    "reject_near_any_opponent",
    "reject_near_armed_opponent",
    "reject_predicted_branch_threat",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def frozen_files(lineup_names=CORE_LINEUP_NAMES) -> dict[str, str]:
    agents = {CANDIDATE}
    for lineup in lineup_names:
        agents.update(LINEUPS[lineup])
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


def protocol(lineup_names=CORE_LINEUP_NAMES, *, robustness=False) -> dict:
    payload = {
        "experiment": (
            "E2 contextual hybrid robustness extension" if robustness
            else "E2 contextual hybrid development screen"
        ),
        "status": (
            "auxiliary robustness only; reuses viewed seeds 66001-66040" if robustness
            else "development only; seeds were viewed in notebook 13"
        ),
        "candidate": CANDIDATE,
        "checkpoint_sha256": digest(ROOT / "agent_code" / CANDIDATE / "dueling_ddqn_model.pt"),
        "scenario": "classic",
        "variants": {
            "one_branch_relaxed": "allow BOMB iff at least one first-step escape branch survives",
            "original_hybrid": (
                "original rule: >=2, or one productive branch through step 120 with no active "
                "bomb/explosion and nearest opponent distance >=5"
            ),
            "phase_productive_hybrid": (
                ">=2, or one productive branch while crates remain, with no active bomb/explosion "
                "and nearest opponent distance >=5; no fixed step cutoff"
            ),
            "armed_threat_hybrid": (
                ">=2, or one productive branch through step 120 unless an active bomb/explosion, "
                "an armed opponent within distance 5, or a hypothetical armed-opponent blast "
                "covers the current/sole first-step tile"
            ),
        },
        "strict_two_branch_status": "archived; not rerun and not a candidate",
        "lineups": {name: LINEUPS[name] for name in lineup_names},
        "seeds": list(SEEDS),
        "rounds_per_seed": ROUNDS_PER_SEED,
        "total_rounds": len(VARIANTS) * len(lineup_names) * len(SEEDS) * ROUNDS_PER_SEED,
        "training": False,
        "exploration": 0.0,
        "pairing_unit": "lineup/seed cluster; average five rounds before inference",
        "contrasts": [f"{left}-{right}" for left, right in CONTRASTS],
        "bootstrap": {"unit": "paired seed cluster", "replicates": 10000, "rng_seed": 66199},
        "primary_metric": "mean_score",
        "trace_version": 2,
        "trace_fields": {
            "decision": [
                "Q margin", "branch directions", "phase", "crate/opponent yield",
                "bomb/explosion context", "opponent proximity/arming", "dynamic branch threat",
                "all four counterfactual permissions", "multi-label rejection reasons",
            ],
            "bomb_lifecycle": [
                "placement context", "per-step viable continuations", "first trapped step",
                "death cause", "score/resources at placement and round end",
            ],
        },
        "selection_rule": (
            "Use the Pareto frontier. A candidate must improve score over original hybrid without "
            "a clear suicide/survival regression, retain lower self-trap than relaxed, and remain "
            "below 500 ms p95. At most one candidate advances to a fresh 40-seed holdout."
        ),
        "implementation_sha256": digest(Path(__file__)),
        "engine_files": {
            name: digest(ROOT / name)
            for name in ("main.py", "environment.py", "agents.py", "settings.py", "events.py", "items.py")
        },
        "frozen_files": frozen_files(lineup_names),
    }
    if robustness:
        if not PROTOCOL_PATH.exists():
            raise ValueError("Core protocol must be locked before the robustness extension")
        payload["parent_protocol"] = str(PROTOCOL_PATH.relative_to(ROOT))
        payload["parent_protocol_sha256"] = digest(PROTOCOL_PATH)
    return payload


def protocol_path_for(lineup: str) -> Path:
    return ROBUSTNESS_PROTOCOL_PATH if lineup in ROBUSTNESS_LINEUP_NAMES else PROTOCOL_PATH


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


def add(position, delta):
    return position[0] + delta[0], position[1] + delta[1]


def bomb_escape_first_steps(extractor, state: dict) -> list[str]:
    """Return independent survivable first moves after placing a bomb."""
    if not extractor.legal_action_mask(state)[ACTIONS.index("BOMB")]:
        return []
    field = state["field"]
    explosion_map = state["explosion_map"]
    start = tuple(state["self"][-1])
    opponents = {tuple(other[-1]) for other in state["others"]}
    schedule = extractor._bomb_schedule(state, add_own_bomb=True)
    result = []
    for action in CARDINAL_ACTIONS:
        target = add(start, MOVE_DELTAS[action])
        if field[target] != 0 or target in opponents:
            continue
        if extractor._bomb_blocks(target, 2, schedule) or extractor._is_dangerous(target, 2, schedule, explosion_map):
            continue
        positions = {target}
        for time_step in range(3, extractor.horizon + 1):
            next_positions = set()
            for position in positions:
                for delta in tuple(MOVE_DELTAS[a] for a in CARDINAL_ACTIONS) + ((0, 0),):
                    next_target = add(position, delta)
                    if field[next_target] != 0 or next_target in opponents:
                        continue
                    if extractor._bomb_blocks(next_target, time_step, schedule):
                        continue
                    if extractor._is_dangerous(next_target, time_step, schedule, explosion_map):
                        continue
                    next_positions.add(next_target)
            positions = next_positions
            if not positions:
                break
        if positions:
            result.append(action)
    return result


def decision_context(extractor, state: dict, survival: np.ndarray | None = None) -> dict:
    position = tuple(state["self"][-1])
    field = state["field"]
    first_steps = bomb_escape_first_steps(extractor, state)
    branch_count = len(first_steps)
    blast = extractor.blast_tiles(field, position)
    crate_yield = int(sum(field[tile] == 1 for tile in blast))
    opponents = list(state["others"])
    opponent_yield = int(sum(tuple(other[-1]) in blast for other in opponents))
    distances = [
        abs(position[0] - int(other[-1][0])) + abs(position[1] - int(other[-1][1]))
        for other in opponents
    ]
    armed_distances = [
        distance for distance, other in zip(distances, opponents) if bool(other[2])
    ]
    nearest = min(distances, default=99)
    nearest_armed = min(armed_distances, default=99)
    active_bombs = len(state["bombs"])
    active_explosion = bool(np.any(state["explosion_map"] > 0))
    sole_target = (
        add(position, MOVE_DELTAS[first_steps[0]]) if branch_count == 1 else None
    )
    predicted_branch_threat = False
    for other in opponents:
        if not bool(other[2]):
            continue
        hypothetical_blast = extractor.blast_tiles(field, tuple(other[-1]))
        if position in hypothetical_blast or (sole_target is not None and sole_target in hypothetical_blast):
            predicted_branch_threat = True
            break
    crates_remaining = int(np.sum(field == 1))
    if crates_remaining:
        phase = "crate_economy"
    elif state["coins"]:
        phase = "coin_only"
    elif state["others"]:
        phase = "hunt"
    else:
        phase = "terminal"
    if survival is None:
        _, survival = base_e2.immediate_survival_mask(extractor, state)
    return {
        "position": [int(position[0]), int(position[1])],
        "phase": phase,
        "crates_remaining": crates_remaining,
        "crate_yield": crate_yield,
        "opponent_yield": opponent_yield,
        "bomb_escape_first_steps": first_steps,
        "bomb_branch_count": branch_count,
        "bomb_survival_count": int(survival[ACTIONS.index("BOMB")]),
        "active_bomb_count": active_bombs,
        "active_explosion": active_explosion,
        "nearest_opponent_distance": int(nearest),
        "nearest_armed_opponent_distance": int(nearest_armed),
        "nearby_armed_opponent": bool(nearest_armed <= NEAR_DISTANCE),
        "predicted_branch_threat": bool(predicted_branch_threat),
        "current_danger": float(extractor._danger_score(state, position)),
        "local_freedom": float(extractor._local_freedom(state, position)),
    }


def permissions(context: dict, passive_bomb_allowed: bool, step: int) -> dict[str, bool]:
    branches = context["bomb_branch_count"]
    productive = context["crate_yield"] > 0
    quiet = context["active_bomb_count"] == 0 and not context["active_explosion"]
    result = {
        "one_branch_relaxed": bool(passive_bomb_allowed and branches >= 1),
        "original_hybrid": bool(
            passive_bomb_allowed
            and (
                branches >= 2
                or (
                    branches == 1 and productive and step <= ORIGINAL_MAX_STEP and quiet
                    and context["nearest_opponent_distance"] >= NEAR_DISTANCE
                )
            )
        ),
        "phase_productive_hybrid": bool(
            passive_bomb_allowed
            and (
                branches >= 2
                or (
                    branches == 1 and context["crates_remaining"] > 0 and productive and quiet
                    and context["nearest_opponent_distance"] >= NEAR_DISTANCE
                )
            )
        ),
        "armed_threat_hybrid": bool(
            passive_bomb_allowed
            and (
                branches >= 2
                or (
                    branches == 1 and productive and step <= ORIGINAL_MAX_STEP
                    and context["active_bomb_count"] == 0 and not context["active_explosion"]
                    and not context["nearby_armed_opponent"]
                    and not context["predicted_branch_threat"]
                )
            )
        ),
    }
    return result


def rejection_reasons(context: dict, step: int) -> dict[str, bool]:
    return {
        "reject_no_escape_branch": context["bomb_branch_count"] < 1,
        "reject_nonproductive": context["crate_yield"] <= 0,
        "reject_no_crates_remaining": context["crates_remaining"] <= 0,
        "reject_step_cutoff": step > ORIGINAL_MAX_STEP,
        "reject_active_bomb": context["active_bomb_count"] > 0,
        "reject_active_explosion": context["active_explosion"],
        "reject_near_any_opponent": context["nearest_opponent_distance"] < NEAR_DISTANCE,
        "reject_near_armed_opponent": context["nearby_armed_opponent"],
        "reject_predicted_branch_threat": context["predicted_branch_threat"],
    }


def make_act(callbacks, torch, variant: str, trace_rows: list):
    def act(self, game_state: dict) -> str:
        proposal_allowed, survival = base_e2.immediate_survival_mask(self.feature_extractor, game_state)
        context = decision_context(self.feature_extractor, game_state, survival)
        all_permissions = permissions(
            context,
            bool(proposal_allowed[ACTIONS.index("BOMB")]),
            int(game_state["step"]),
        )
        allowed = proposal_allowed.copy()
        allowed[ACTIONS.index("BOMB")] = all_permissions[variant]
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
        hunt_phase = context["phase"] == "hunt"
        if hunt_phase or not allowed[ACTIONS.index("BOMB")]:
            for index, candidate_action in enumerate(ACTIONS):
                if candidate_action == "BOMB" or action_features[index, danger_index] > 0.0:
                    continue
                target = add(position, callbacks.MOVE_DELTAS[candidate_action])
                revisit_penalties[index] = callbacks.REVISIT_Q_PENALTY * min(
                    self.recent_positions.count(target), 3
                )
        self.recent_positions.append(position)
        state = torch.from_numpy(action_features.astype(np.float32).reshape(-1)).unsqueeze(0).to(self.device)
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
        proposal_best = np.flatnonzero(np.isclose(proposal_q, np.max(proposal_q)))
        bomb_index = ACTIONS.index("BOMB")
        bomb_proposal = bool(bomb_index in proposal_best)
        masked_q = np.where(allowed, q_values, -np.inf)
        best = np.flatnonzero(np.isclose(masked_q, np.max(masked_q)))
        chosen = ACTIONS[self.rng.choice(best.tolist())]
        finite_nonbomb = q_values[np.asarray([allowed[i] and i != bomb_index for i in range(len(ACTIONS))])]
        best_nonbomb_q = float(np.max(finite_nonbomb)) if finite_nonbomb.size else math.nan
        rejected = bool(bomb_proposal and not all_permissions[variant])
        reasons = rejection_reasons(context, int(game_state["step"]))
        trace_rows.append(
            {
                "round": int(game_state["round"]), "step": int(game_state["step"]),
                "variant": variant, "action": chosen, "bomb_proposal": bomb_proposal,
                "bomb_permitted": bool(all_permissions[variant]), "bomb_rejected": rejected,
                "bomb_q": float(q_values[bomb_index]), "best_nonbomb_q": best_nonbomb_q,
                "bomb_q_margin": float(q_values[bomb_index] - best_nonbomb_q) if np.isfinite(best_nonbomb_q) else math.nan,
                **context,
                **{f"allowed_by_{name}": allowed_value for name, allowed_value in all_permissions.items()},
                **{key: bool(rejected and value) for key, value in reasons.items()},
                "decision_ms": None,
            }
        )
        return chosen
    return act


def run_one(variant: str, lineup: str, seed: int) -> None:
    import torch
    import main
    import events as event_names
    from agents import Agent
    from environment import BombeRLeWorld

    locked_protocol_path = protocol_path_for(lineup)
    spec = json.loads(locked_protocol_path.read_text(encoding="utf-8"))
    if lineup not in spec["lineups"]:
        raise ValueError(f"Lineup {lineup} is not present in {locked_protocol_path.name}")
    validate_frozen_files(spec)
    raw_path, trace_path, manifest_path = paths(lineup, variant, seed)
    if any(path.exists() for path in (raw_path, trace_path, manifest_path)):
        raise FileExistsError(f"Existing or partial run: {lineup}/{variant}/{seed}")
    for path in (raw_path, trace_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = importlib.import_module(f"agent_code.{CANDIDATE}.callbacks")
    decisions: list[dict] = []
    bomb_rows: list[dict] = []
    callbacks.act = make_act(callbacks, torch, variant, decisions)
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    original_setup = BombeRLeWorld.setup_agents
    original_new = BombeRLeWorld.new_round
    original_poll = BombeRLeWorld.poll_and_run_agents
    original_action = BombeRLeWorld.perform_agent_action
    original_evaluate = BombeRLeWorld.evaluate_explosions
    original_end_round = BombeRLeWorld.end_round
    original_wait = Agent.wait_for_act
    current = {"candidate": None, "extractor": None, "initial": [], "pending": None, "bomb_counter": 0}

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
        current["initial"].append(
            {"round": int(world.round), "round_seed": int(round_seed),
             "sha256": hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest()}
        )

    def wait_for_act(agent):
        action, think_time = original_wait(agent)
        if agent is current["candidate"] and decisions:
            decisions[-1]["decision_ms"] = 1000.0 * float(think_time)
        return action, think_time

    def perform_agent_action(world, agent, action):
        successfully_armed = agent is current["candidate"] and action == "BOMB" and agent.bombs_left
        result = original_action(world, agent, action)
        if successfully_armed:
            current["bomb_counter"] += 1
            decision = decisions[-1]
            bomb = {
                "bomb_id": f"{lineup}:{variant}:{seed}:{world.round}:{current['bomb_counter']}",
                "round": int(world.round), "placed_step": int(world.step),
                "origin": [int(agent.x), int(agent.y)], "phase": decision["phase"],
                "branch_count": decision["bomb_branch_count"],
                "first_steps": decision["bomb_escape_first_steps"],
                "crate_yield": decision["crate_yield"], "opponent_yield": decision["opponent_yield"],
                "bomb_q_margin": decision["bomb_q_margin"],
                "score_at_placement": int(agent.score),
                "coins_at_placement": int(agent.statistics["coins"]),
                "crates_at_placement": int(agent.statistics["crates"]),
                "kills_at_placement": int(agent.statistics["kills"]),
                "viability_trace": [], "self_trapped_after_bomb": False,
                "first_trapped_step": None, "death_cause": None, "death_step": None,
                "score_at_round_end": None, "coins_at_round_end": None,
                "crates_at_round_end": None, "kills_at_round_end": None, "survived_round": None,
            }
            bomb_rows.append(bomb)
            current["pending"] = bomb
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
                legal = current["extractor"].legal_action_mask(state)
                viable_actions = [
                    action for index, action in enumerate(ACTIONS)
                    if legal[index] and current["extractor"].survival_count(state, action) > 0
                ]
                pending["viability_trace"].append(
                    {"step": int(world.step), "viable_action_count": len(viable_actions), "viable_actions": viable_actions}
                )
                if not viable_actions and not pending["self_trapped_after_bomb"]:
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
        if was_alive and candidate not in world.active_agents and current["pending"] is not None:
            own = any(owner is candidate for owner in owners)
            other = any(owner is not candidate for owner in owners)
            current["pending"]["death_cause"] = (
                "own_and_opponent_bomb" if own and other else
                "own_bomb_suicide" if own else
                "opponent_bomb" if other else "other"
            )
            current["pending"]["death_step"] = int(world.step)
        return result

    def end_round(world):
        candidate = current["candidate"]
        for bomb in [row for row in bomb_rows if row["round"] == int(world.round)]:
            bomb["score_at_round_end"] = int(candidate.score)
            bomb["coins_at_round_end"] = int(candidate.statistics["coins"])
            bomb["crates_at_round_end"] = int(candidate.statistics["crates"])
            bomb["kills_at_round_end"] = int(candidate.statistics["kills"])
            bomb["survived_round"] = bool(not candidate.dead)
        return original_end_round(world)

    BombeRLeWorld.setup_agents = setup_agents
    BombeRLeWorld.new_round = new_round
    BombeRLeWorld.poll_and_run_agents = poll_and_run_agents
    BombeRLeWorld.perform_agent_action = perform_agent_action
    BombeRLeWorld.evaluate_explosions = evaluate_explosions
    BombeRLeWorld.end_round = end_round
    Agent.wait_for_act = wait_for_act
    command = [
        "main.py", "play", "--no-gui", "--agents", CANDIDATE, *LINEUPS[lineup],
        "--scenario", "classic", "--n-rounds", str(ROUNDS_PER_SEED), "--seed", str(seed),
        "--save-stats", str(raw_path), "--match-name", f"e2ctx_{lineup}_{variant}_{seed}",
    ]
    main.main(command[1:])
    atomic_json(trace_path, {"trace_version": 2, "decisions": decisions, "bombs": bomb_rows})
    atomic_json(
        manifest_path,
        {
            "variant": variant, "lineup": lineup, "seed": seed, "rounds": ROUNDS_PER_SEED,
            "training": False, "command": command, "initial_conditions": current["initial"],
            "raw_stats": str(raw_path.relative_to(ROOT)), "raw_sha256": digest(raw_path),
            "trace": str(trace_path.relative_to(ROOT)), "trace_sha256": digest(trace_path),
            "protocol": str(locked_protocol_path.relative_to(ROOT)),
            "protocol_sha256": digest(locked_protocol_path),
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
        rd = decisions[round_id]
        rb = bombs[round_id]
        times = [float(row["decision_ms"]) for row in rd if row["decision_ms"] is not None]
        raw_trapped = [row for row in rb if row["self_trapped_after_bomb"]]
        escape_collapses = [row for row in rb if strict_trap_flags(row)["post_placement_escape_collapse"]]
        confirmed_traps = [row for row in rb if strict_trap_flags(row)["confirmed_catastrophic_self_trap"]]
        rejected = [row for row in rd if row["bomb_rejected"]]
        productive_single = [
            row for row in rd
            if row["bomb_proposal"] and row["bomb_branch_count"] == 1 and row["crate_yield"] > 0
        ]
        row = {
            "variant": variant, "lineup": lineup, "seed": seed, "round": round_id,
            "score": float(outcome["score"]), "coins": float(outcome["coins"]),
            "crates": float(outcome["crates"]), "bombs": float(outcome["bombs"]),
            "kills": float(outcome["kills"]), "suicides": float(outcome["suicides"]),
            "invalid": float(outcome["invalid"]), "survived": float(bool(outcome["survived"])),
            "bomb_proposals": float(sum(item["bomb_proposal"] for item in rd)),
            "bomb_rejections": float(len(rejected)), "actual_bombs": float(len(rb)),
            # Retain the original broad signal for auditability, but make the
            # post-placement and outcome-confirmed definitions explicit.
            "self_trapped_after_bomb": float(len(raw_trapped)),
            "post_placement_escape_collapse": float(len(escape_collapses)),
            "confirmed_catastrophic_self_trap": float(len(confirmed_traps)),
            "same_step_only_zero": float(len(raw_trapped) - len(escape_collapses)),
            "self_trap_to_suicide": float(sum(item["death_cause"] in {"own_bomb_suicide", "own_and_opponent_bomb"} for item in raw_trapped)),
            "self_trap_to_any_death": float(sum(item["death_cause"] is not None for item in raw_trapped)),
            "economic_self_traps": float(sum(item["phase"] == "crate_economy" for item in raw_trapped)),
            "hunt_self_traps": float(sum(item["phase"] == "hunt" for item in raw_trapped)),
            "economic_confirmed_self_traps": float(sum(item["phase"] == "crate_economy" for item in confirmed_traps)),
            "hunt_confirmed_self_traps": float(sum(item["phase"] == "hunt" for item in confirmed_traps)),
            "productive_single_branch_proposals": float(len(productive_single)),
            "productive_single_branch_accepts": float(sum(item["bomb_permitted"] for item in productive_single)),
            "decision_ms_mean": float(np.mean(times)) if times else math.nan,
            "decision_ms_p95": float(np.percentile(times, 95)) if times else math.nan,
        }
        for key in REJECTION_KEYS:
            row[key] = float(sum(item[key] for item in rejected))
        rows.append(row)
    return rows


OWN_BOMB_DEATH_CAUSES = {"own_bomb_suicide", "own_and_opponent_bomb"}


def strict_trap_flags(bomb: dict) -> dict:
    """Derive stricter trap labels from an immutable bomb lifecycle trace.

    The old flag is raised by any zero-viability sample, including the sample
    taken immediately after the BOMB action.  That sample is not a subsequent
    decision opportunity.  The strict mechanism signal therefore requires a
    zero-viability sample at a later step.  The primary catastrophic outcome
    additionally requires the candidate's own bomb to be causally involved in
    its death.  This deliberately favours specificity over sensitivity.
    """
    placed_step = int(bomb["placed_step"])
    later_zero_steps = [
        int(sample["step"])
        for sample in bomb.get("viability_trace", [])
        if int(sample["step"]) > placed_step and int(sample["viable_action_count"]) == 0
    ]
    collapse = bool(later_zero_steps)
    confirmed = collapse and bomb.get("death_cause") in OWN_BOMB_DEATH_CAUSES
    return {
        "raw_zero": bool(bomb.get("self_trapped_after_bomb")),
        "post_placement_escape_collapse": collapse,
        "confirmed_catastrophic_self_trap": confirmed,
        "first_post_placement_zero_step": min(later_zero_steps) if later_zero_steps else None,
    }


def nanmean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else math.nan


def summarize() -> dict:
    core_spec = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    robustness_spec = json.loads(ROBUSTNESS_PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_frozen_files(core_spec)
    validate_frozen_files(robustness_spec)
    spec = dict(core_spec)
    spec.update({
        "experiment": "E2 contextual hybrid development screen plus robustness extension",
        "status": "development/auxiliary robustness only; all lineups reuse viewed seeds 66001-66040",
        "lineups": {**core_spec["lineups"], **robustness_spec["lineups"]},
        "core_lineups": list(CORE_LINEUP_NAMES),
        "robustness_lineups": list(ROBUSTNESS_LINEUP_NAMES),
        "core_rounds": core_spec["total_rounds"],
        "robustness_extension_rounds": robustness_spec["total_rounds"],
        "total_rounds": core_spec["total_rounds"] + robustness_spec["total_rounds"],
        "protocol_sources": {
            str(PROTOCOL_PATH.relative_to(ROOT)): digest(PROTOCOL_PATH),
            str(ROBUSTNESS_PROTOCOL_PATH.relative_to(ROOT)): digest(ROBUSTNESS_PROTOCOL_PATH),
        },
        "frozen_files": {**core_spec["frozen_files"], **robustness_spec["frozen_files"]},
    })
    rows, ledger = [], []
    for lineup in LINEUPS:
        for variant in VARIANTS:
            for seed in SEEDS:
                raw_path, trace_path, manifest_path = paths(lineup, variant, seed)
                if not all(path.exists() for path in (raw_path, trace_path, manifest_path)):
                    raise ValueError(f"Missing run: {lineup}/{variant}/{seed}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                locked_protocol_path = protocol_path_for(lineup)
                if manifest["protocol_sha256"] != digest(locked_protocol_path):
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
    reason_totals = []
    strict_attribution = []
    for lineup in LINEUPS:
        for variant in VARIANTS:
            group = [row for row in rows if row["lineup"] == lineup and row["variant"] == variant]
            all_times = []
            lifecycle_bombs = []
            raw_trap_steps = []
            confirmed_trap_steps = []
            for seed in SEEDS:
                trace = json.loads(paths(lineup, variant, seed)[1].read_text(encoding="utf-8"))
                all_times.extend(float(row["decision_ms"]) for row in trace["decisions"] if row["decision_ms"] is not None)
                lifecycle_bombs.extend(trace["bombs"])
                raw_trap_steps.extend(row["placed_step"] for row in trace["bombs"] if row["self_trapped_after_bomb"])
                confirmed_trap_steps.extend(
                    strict_trap_flags(row)["first_post_placement_zero_step"]
                    for row in trace["bombs"]
                    if strict_trap_flags(row)["confirmed_catastrophic_self_trap"]
                )
            totals = {metric: float(sum(row[metric] for row in group)) for metric in metric_names if not metric.startswith("decision_ms")}
            aggregate.append({
                "lineup": lineup, "variant": variant, "seed_clusters": len(SEEDS), "rounds": len(group),
                **{metric: nanmean([row[metric] for row in group]) for metric in metric_names if metric != "decision_ms_p95"},
                "decision_ms_p95": float(np.percentile(all_times, 95)),
                "bomb_proposals_total": int(totals["bomb_proposals"]),
                "bomb_rejections_total": int(totals["bomb_rejections"]),
                "actual_bombs_total": int(totals["actual_bombs"]),
                "self_trapped_total": int(totals["self_trapped_after_bomb"]),
                "self_traps_per_100_bombs": 100.0 * totals["self_trapped_after_bomb"] / max(totals["actual_bombs"], 1),
                "post_placement_escape_collapse_total": int(totals["post_placement_escape_collapse"]),
                "post_placement_escape_collapses_per_100_bombs": 100.0 * totals["post_placement_escape_collapse"] / max(totals["actual_bombs"], 1),
                "confirmed_catastrophic_self_trap_total": int(totals["confirmed_catastrophic_self_trap"]),
                "confirmed_catastrophic_self_traps_per_100_bombs": 100.0 * totals["confirmed_catastrophic_self_trap"] / max(totals["actual_bombs"], 1),
                "same_step_only_zero_total": int(totals["same_step_only_zero"]),
                "self_trap_to_suicide_rate": totals["self_trap_to_suicide"] / max(totals["self_trapped_after_bomb"], 1),
                "self_trap_to_any_death_rate": totals["self_trap_to_any_death"] / max(totals["self_trapped_after_bomb"], 1),
                "median_trap_step": float(np.median(raw_trap_steps)) if raw_trap_steps else math.nan,
                "median_confirmed_trap_step": float(np.median(confirmed_trap_steps)) if confirmed_trap_steps else math.nan,
            })
            reason_totals.append({
                "lineup": lineup, "variant": variant,
                **{key: int(totals[key]) for key in REJECTION_KEYS},
            })
            raw_bombs = [row for row in lifecycle_bombs if strict_trap_flags(row)["raw_zero"]]
            collapsed_bombs = [
                row for row in lifecycle_bombs
                if strict_trap_flags(row)["post_placement_escape_collapse"]
            ]
            confirmed_bombs = [
                row for row in lifecycle_bombs
                if strict_trap_flags(row)["confirmed_catastrophic_self_trap"]
            ]
            placement_scores = [float(row["score_at_placement"]) for row in confirmed_bombs]
            score_gains = [
                float(row["score_at_round_end"] - row["score_at_placement"])
                for row in confirmed_bombs
            ]
            strict_attribution.append({
                "lineup": lineup,
                "variant": variant,
                "actual_bombs_total": len(lifecycle_bombs),
                "raw_zero_total": len(raw_bombs),
                "same_step_only_zero_total": len(raw_bombs) - len(collapsed_bombs),
                "post_placement_escape_collapse_total": len(collapsed_bombs),
                "confirmed_catastrophic_self_trap_total": len(confirmed_bombs),
                "opponent_only_death_after_collapse_total": sum(
                    row.get("death_cause") == "opponent_bomb" for row in collapsed_bombs
                ),
                "no_linked_death_after_collapse_total": sum(
                    row.get("death_cause") is None for row in collapsed_bombs
                ),
                "confirmed_crate_economy_total": sum(
                    row["phase"] == "crate_economy" for row in confirmed_bombs
                ),
                "confirmed_coin_only_total": sum(
                    row["phase"] == "coin_only" for row in confirmed_bombs
                ),
                "confirmed_hunt_total": sum(row["phase"] == "hunt" for row in confirmed_bombs),
                "confirmed_placement_step_median": (
                    float(np.median([row["placed_step"] for row in confirmed_bombs]))
                    if confirmed_bombs else math.nan
                ),
                "confirmed_trap_step_median": (
                    float(np.median([
                        strict_trap_flags(row)["first_post_placement_zero_step"]
                        for row in confirmed_bombs
                    ])) if confirmed_bombs else math.nan
                ),
                "confirmed_death_latency_median": (
                    float(np.median([
                        row["death_step"] - strict_trap_flags(row)["first_post_placement_zero_step"]
                        for row in confirmed_bombs
                    ])) if confirmed_bombs else math.nan
                ),
                "score_at_placement_mean": float(np.mean(placement_scores)) if placement_scores else math.nan,
                "score_at_placement_median": float(np.median(placement_scores)) if placement_scores else math.nan,
                "score_at_placement_ge_3_total": sum(value >= 3 for value in placement_scores),
                "score_at_placement_ge_5_total": sum(value >= 5 for value in placement_scores),
                "no_score_gain_after_confirmed_total": sum(value == 0 for value in score_gains),
                "no_score_gain_after_confirmed_rate": (
                    float(np.mean(np.asarray(score_gains) == 0)) if score_gains else math.nan
                ),
                "score_gain_after_confirmed_mean": float(np.mean(score_gains)) if score_gains else math.nan,
            })
    comparison_metrics = [
        "score", "coins", "crates", "bombs", "kills", "suicides", "survived",
        "bomb_proposals", "bomb_rejections", "actual_bombs", "self_trapped_after_bomb",
        "post_placement_escape_collapse", "confirmed_catastrophic_self_trap", "same_step_only_zero",
        "self_trap_to_suicide", "self_trap_to_any_death", "economic_self_traps",
        "hunt_self_traps", "economic_confirmed_self_traps", "hunt_confirmed_self_traps",
        "decision_ms_mean",
    ]
    comparisons = []
    rng = np.random.default_rng(spec["bootstrap"]["rng_seed"])
    for lineup in LINEUPS:
        for left_name, right_name in CONTRASTS:
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
                    "ci_low": float(np.percentile(samples, 2.5)), "ci_high": float(np.percentile(samples, 97.5)),
                    "positive_seeds": int(np.sum(observed > 0)), "zero_seeds": int(np.sum(observed == 0)),
                    "negative_seeds": int(np.sum(observed < 0)), "paired_seeds": int(len(observed)),
                })
    payload = {
        "self_trap_definitions": {
            "raw_zero": "Any zero-viability lifecycle sample, including the BOMB placement step; retained only as a diagnostic.",
            "post_placement_escape_collapse": "At least one zero-viability sample at a step strictly after BOMB placement.",
            "confirmed_catastrophic_self_trap": "Post-placement escape collapse followed by death in which the placed agent's own bomb was causally involved; primary strict metric.",
        },
        "protocol": spec, "ledger": ledger, "round_rows": rows, "seed_rows": seed_rows,
        "aggregate": aggregate, "rejection_reason_totals": reason_totals,
        "strict_self_trap_attribution": strict_attribution,
        "paired_bootstrap": comparisons,
    }
    atomic_json(SUMMARY_PATH, payload)
    atomic_csv(STRICT_ATTRIBUTION_CSV, strict_attribution)
    atomic_csv(STRICT_ROUND_ROWS_CSV, [{
        key: row[key] for key in (
            "lineup", "variant", "seed", "round", "score", "suicides", "survived",
            "actual_bombs", "self_trapped_after_bomb", "same_step_only_zero",
            "post_placement_escape_collapse", "confirmed_catastrophic_self_trap",
            "economic_confirmed_self_traps", "hunt_confirmed_self_traps",
        )
    } for row in rows])
    atomic_csv(STRICT_BCI_CSV, [
        row for row in comparisons
        if row["metric"] in {
            "self_trapped_after_bomb", "same_step_only_zero",
            "post_placement_escape_collapse", "confirmed_catastrophic_self_trap",
        }
    ])
    return payload


def status() -> dict:
    complete, partial, missing = 0, [], []
    for lineup in LINEUPS:
        for variant in VARIANTS:
            for seed in SEEDS:
                run_paths = paths(lineup, variant, seed)
                count = sum(path.exists() for path in run_paths)
                if count == len(run_paths): complete += 1
                elif count: partial.append({"lineup": lineup, "variant": variant, "seed": seed})
                else: missing.append({"lineup": lineup, "variant": variant, "seed": seed})
    return {"complete_clusters": complete, "total_clusters": len(LINEUPS) * len(VARIANTS) * len(SEEDS), "partial": partial, "missing": missing}


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
        if not PROTOCOL_PATH.exists():
            raise ValueError("The immutable core protocol is missing")
        validate_frozen_files(json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")))
        robustness_spec = protocol(ROBUSTNESS_LINEUP_NAMES, robustness=True)
        if (
            ROBUSTNESS_PROTOCOL_PATH.exists()
            and json.loads(ROBUSTNESS_PROTOCOL_PATH.read_text(encoding="utf-8")) != robustness_spec
        ):
            raise ValueError("Contextual hybrid robustness protocol already locked and differs")
        atomic_json(ROBUSTNESS_PROTOCOL_PATH, robustness_spec)
    elif args.command == "run":
        if args.seed not in SEEDS: raise ValueError("Seed outside locked development set")
        run_one(args.variant, args.lineup, args.seed)
    elif args.command == "summarize": summarize()
    else: print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    main_cli()
