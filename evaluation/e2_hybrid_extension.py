"""Append-only E2 extension adding the original hybrid bomb rule.

The original two E2 arms and their locked protocol are never rewritten.  This
module runs the unmodified ``dueling_hunt_center_04`` hybrid permission rule on
the same seeds and lineups, then builds a three-arm summary with all pairwise
paired bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation import e2_bomb_escape_branches as e2


ROOT = e2.ROOT
RESULT_ROOT = e2.RESULT_ROOT
VARIANT = "original_hybrid"
ALL_VARIANTS = ("two_branch_final", VARIANT, "one_branch_relaxed")
CONTRASTS = (
    ("two_branch_final", "one_branch_relaxed"),
    ("two_branch_final", VARIANT),
    (VARIANT, "one_branch_relaxed"),
)
EXTENSION_PROTOCOL = RESULT_ROOT / "protocol_hybrid_extension.json"
THREEWAY_SUMMARY = RESULT_ROOT / "summary_threeway.json"


def extension_protocol() -> dict:
    base = RESULT_ROOT / "protocol.json"
    spec = json.loads(base.read_text(encoding="utf-8"))
    return {
        "experiment": "E2 append-only original-hybrid extension",
        "base_protocol": str(base.relative_to(ROOT)),
        "base_protocol_sha256": e2.digest(base),
        "added_variant": {
            "name": VARIANT,
            "permission_rule": (
                "original FeatureExtractor.safe_action_mask: normally >=2 branches; "
                "allow one branch only for a productive quiet-opening bomb through step 120, "
                "with no other bomb/explosion and nearest opponent at least 5 tiles away"
            ),
        },
        "candidate": spec["candidate"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "lineups": spec["lineups"],
        "seeds": spec["seeds"],
        "rounds_per_seed": spec["rounds_per_seed"],
        "added_rounds": len(spec["lineups"]) * len(e2.SEEDS) * e2.ROUNDS_PER_SEED,
        "three_arm_total_rounds": 3 * len(spec["lineups"]) * len(e2.SEEDS) * e2.ROUNDS_PER_SEED,
        "training": False,
        "exploration": 0.0,
        "pairing_unit": spec["pairing_unit"],
        "bootstrap": spec["bootstrap"],
        "pipeline_note": (
            "Same instrumented inference pipeline, candidate, lineups, seeds, round seeding, "
            "immediate-lethality filter, Q adjustments, tracing, and engine as the base E2. "
            "Only final BOMB permission uses the original hybrid safe_action_mask."
        ),
        "implementation_sha256": e2.digest(Path(__file__)),
        "frozen_files": spec["frozen_files"],
    }


def prepare() -> None:
    spec = extension_protocol()
    if EXTENSION_PROTOCOL.exists():
        existing = json.loads(EXTENSION_PROTOCOL.read_text(encoding="utf-8"))
        if existing != spec:
            raise ValueError("Hybrid extension protocol is already locked and differs")
    e2.atomic_json(EXTENSION_PROTOCOL, spec)


def make_hybrid_act(callbacks, torch, _unused, trace_rows: list):
    """Exact frozen callback with its original hybrid safe mask plus E2 tracing."""

    def act(self, game_state: dict) -> str:
        proposal_allowed, survival = e2.immediate_survival_mask(
            self.feature_extractor, game_state
        )
        branch_count = self.feature_extractor.bomb_escape_branch_count(game_state)
        allowed = self.feature_extractor.safe_action_mask(game_state)
        allowed_indices = np.flatnonzero(allowed)
        if allowed_indices.size == 0:
            allowed[e2.ACTIONS.index("WAIT")] = True
            allowed_indices = np.flatnonzero(allowed)

        action_features = self.feature_extractor.action_features(game_state)
        position = tuple(game_state["self"][-1])
        if self.position_round != game_state["round"]:
            self.recent_positions.clear()
            self.position_round = game_state["round"]
        revisit_penalties = np.zeros(len(e2.ACTIONS), dtype=np.float32)
        danger_index = callbacks.FEATURE_NAMES.index("target_danger")
        hunt_phase = (
            not game_state["coins"]
            and not np.any(game_state["field"] == 1)
            and bool(game_state["others"])
        )
        bomb_index = e2.ACTIONS.index("BOMB")
        if hunt_phase or not allowed[bomb_index]:
            for index, candidate_action in enumerate(e2.ACTIONS):
                if candidate_action == "BOMB" or action_features[index, danger_index] > 0.0:
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
        proposal_best = np.flatnonzero(np.isclose(proposal_q, np.max(proposal_q)))
        bomb_proposal = bool(bomb_index in proposal_best)
        bomb_permitted = bool(allowed[bomb_index])
        masked_q = np.where(allowed, q_values, -np.inf)
        best = np.flatnonzero(np.isclose(masked_q, np.max(masked_q)))
        chosen = e2.ACTIONS[self.rng.choice(best.tolist())]
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


def run_one(lineup: str, seed: int) -> None:
    spec = json.loads(EXTENSION_PROTOCOL.read_text(encoding="utf-8"))
    if e2.digest(RESULT_ROOT / "protocol.json") != spec["base_protocol_sha256"]:
        raise ValueError("Base E2 protocol changed")
    e2.validate_frozen_files(spec)
    e2.VARIANTS[VARIANT] = None
    original_factory = e2.make_instrumented_act
    e2.make_instrumented_act = make_hybrid_act
    try:
        e2.run_one(VARIANT, lineup, seed)
    finally:
        e2.make_instrumented_act = original_factory

    _, _, manifest_path = e2.paths(lineup, VARIANT, seed)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "minimum_branches": "original_hybrid",
            "base_protocol_sha256": spec["base_protocol_sha256"],
            "protocol_sha256": e2.digest(EXTENSION_PROTOCOL),
            "protocol": str(EXTENSION_PROTOCOL.relative_to(ROOT)),
        }
    )
    e2.atomic_json(manifest_path, manifest)


def nanmean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else math.nan


def percentile_interval(observed: np.ndarray, rng, replicates: int):
    indices = rng.integers(0, len(observed), size=(replicates, len(observed)))
    samples = observed[indices].mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def summarize() -> dict:
    # Revalidate and regenerate the untouched two-arm summary first.
    base = e2.summarize()
    extension = json.loads(EXTENSION_PROTOCOL.read_text(encoding="utf-8"))
    e2.validate_frozen_files(extension)
    if e2.digest(RESULT_ROOT / "protocol.json") != extension["base_protocol_sha256"]:
        raise ValueError("Base protocol hash mismatch")

    rows = list(base["round_rows"])
    ledger = list(base["ledger"])
    for lineup in extension["lineups"]:
        for seed in e2.SEEDS:
            raw_path, trace_path, manifest_path = e2.paths(lineup, VARIANT, seed)
            if not all(path.exists() for path in (raw_path, trace_path, manifest_path)):
                raise ValueError(f"Missing hybrid run: {lineup}/{seed}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["protocol_sha256"] != e2.digest(EXTENSION_PROTOCOL):
                raise ValueError(f"Extension protocol mismatch: {manifest_path}")
            if manifest["raw_sha256"] != e2.digest(raw_path) or manifest["trace_sha256"] != e2.digest(trace_path):
                raise ValueError(f"Hybrid result hash mismatch: {manifest_path}")
            ledger.append(
                {
                    "lineup": lineup,
                    "variant": VARIANT,
                    "seed": seed,
                    "manifest": str(manifest_path.relative_to(ROOT)),
                    "raw_sha256": e2.digest(raw_path),
                    "trace_sha256": e2.digest(trace_path),
                }
            )
            rows.extend(e2.round_rows(lineup, VARIANT, seed))

    metric_names = [
        key for key in rows[0]
        if key not in {"variant", "lineup", "seed", "round"}
    ]
    seed_rows = []
    for lineup in extension["lineups"]:
        for variant in ALL_VARIANTS:
            for seed in e2.SEEDS:
                group = [
                    row for row in rows
                    if row["lineup"] == lineup and row["variant"] == variant and row["seed"] == seed
                ]
                seed_rows.append(
                    {
                        "lineup": lineup,
                        "variant": variant,
                        "seed": seed,
                        **{metric: nanmean([row[metric] for row in group]) for metric in metric_names},
                    }
                )

    aggregate = []
    for lineup in extension["lineups"]:
        for variant in ALL_VARIANTS:
            group = [row for row in rows if row["lineup"] == lineup and row["variant"] == variant]
            all_times = []
            for seed in e2.SEEDS:
                trace = json.loads(e2.paths(lineup, variant, seed)[1].read_text(encoding="utf-8"))
                all_times.extend(
                    float(row["decision_ms"])
                    for row in trace["decisions"] if row["decision_ms"] is not None
                )
            aggregate.append(
                {
                    "lineup": lineup,
                    "variant": variant,
                    "seed_clusters": len(e2.SEEDS),
                    "rounds": len(group),
                    **{
                        metric: nanmean([row[metric] for row in group])
                        for metric in metric_names if metric != "decision_ms_p95"
                    },
                    "decision_ms_p95": float(np.percentile(all_times, 95)),
                    "bomb_proposals_total": int(sum(row["bomb_proposals"] for row in group)),
                    "bomb_rejections_total": int(sum(row["bomb_rejections"] for row in group)),
                    "actual_bombs_total": int(sum(row["actual_bombs"] for row in group)),
                    "self_trapped_total": int(sum(row["self_trapped_after_bomb"] for row in group)),
                }
            )

    metrics = [
        "score", "coins", "crates", "bombs", "kills", "suicides", "survived",
        "bomb_proposals", "bomb_rejections", "actual_bombs",
        "self_trapped_after_bomb", "decision_ms_mean",
    ]
    comparisons = []
    rng = np.random.default_rng(extension["bootstrap"]["rng_seed"])
    for lineup in extension["lineups"]:
        for left_name, right_name in CONTRASTS:
            for metric in metrics:
                differences = []
                for seed in e2.SEEDS:
                    left = next(row for row in seed_rows if row["lineup"] == lineup and row["variant"] == left_name and row["seed"] == seed)
                    right = next(row for row in seed_rows if row["lineup"] == lineup and row["variant"] == right_name and row["seed"] == seed)
                    differences.append(left[metric] - right[metric])
                observed = np.asarray(differences, dtype=float)
                observed = observed[~np.isnan(observed)]
                low, high = percentile_interval(observed, rng, extension["bootstrap"]["replicates"])
                comparisons.append(
                    {
                        "lineup": lineup,
                        "contrast": f"{left_name}-{right_name}",
                        "metric": metric,
                        "mean_difference": float(np.mean(observed)),
                        "ci_low": low,
                        "ci_high": high,
                        "positive_seeds": int(np.sum(observed > 0)),
                        "zero_seeds": int(np.sum(observed == 0)),
                        "negative_seeds": int(np.sum(observed < 0)),
                        "paired_seeds": int(len(observed)),
                    }
                )

    combined_protocol = {
        **base["protocol"],
        "variants": {
            **base["protocol"]["variants"],
            VARIANT: extension["added_variant"],
        },
        "total_rounds": extension["three_arm_total_rounds"],
        "extension_protocol": str(EXTENSION_PROTOCOL.relative_to(ROOT)),
        "extension_protocol_sha256": e2.digest(EXTENSION_PROTOCOL),
        "contrasts": [f"{left}-{right}" for left, right in CONTRASTS],
    }
    payload = {
        "protocol": combined_protocol,
        "base_protocol": base["protocol"],
        "extension_protocol": extension,
        "ledger": ledger,
        "round_rows": rows,
        "seed_rows": seed_rows,
        "aggregate": aggregate,
        "paired_bootstrap": comparisons,
    }
    e2.atomic_json(THREEWAY_SUMMARY, payload)
    return payload


def status() -> dict:
    complete = 0
    partial = []
    missing = []
    for lineup in e2.CORE_LINEUPS:
        for seed in e2.SEEDS:
            run_paths = e2.paths(lineup, VARIANT, seed)
            count = sum(path.exists() for path in run_paths)
            if count == len(run_paths):
                complete += 1
            elif count:
                partial.append({"lineup": lineup, "seed": seed})
            else:
                missing.append({"lineup": lineup, "seed": seed})
    return {"complete_clusters": complete, "total_clusters": 80, "partial": partial, "missing": missing}


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--lineup", choices=e2.CORE_LINEUPS, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    sub.add_parser("summarize")
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run":
        if args.seed not in e2.SEEDS:
            raise ValueError("Seed outside locked E2 range")
        run_one(args.lineup, args.seed)
    elif args.command == "summarize":
        summarize()
    else:
        print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    main_cli()
