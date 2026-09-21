"""Summarize the locked paired holdout for engine-aligned checkpoint 0200."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics


LINEUPS = ("rule3", "senior_mixed", "old_feature_rule")
ROUND_METRICS = (
    "score", "coins", "crates", "bombs", "kills", "suicides", "invalid",
    "survived", "win", "rank", "score_margin",
)


def read_rounds(path: Path) -> tuple[list[dict], list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    initial = payload["experiment"]["initial_conditions"]
    rows = []
    for round_data in payload["by_round"].values():
        agents = round_data["agents"]
        candidate = agents["candidate"]
        scores = [float(row["score"]) for row in agents.values()]
        opponent_scores = [float(row["score"]) for name, row in agents.items()
                           if name != "candidate"]
        score = float(candidate["score"])
        rows.append({
            "score": score,
            "coins": float(candidate["coins"]),
            "crates": float(candidate["crates"]),
            "bombs": float(candidate["bombs"]),
            "kills": float(candidate["kills"]),
            "suicides": float(candidate["suicides"]),
            "invalid": float(candidate["invalid"]),
            "survived": float(bool(candidate["survived"])),
            "win": float(score == max(scores)),
            "rank": float(1 + sum(other > score for other in scores)),
            "score_margin": score - statistics.fmean(opponent_scores),
        })
    if len(rows) != len(initial):
        raise ValueError(f"Round/initial-condition mismatch in {path}")
    return rows, initial


def mean_metrics(rows: list[dict]) -> dict:
    return {metric: statistics.fmean(row[metric] for row in rows)
            for metric in ROUND_METRICS}


def percentile(sorted_values: list[float], probability: float) -> float:
    index = probability * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def bootstrap_interval(differences: dict[str, list[float]], rng: random.Random,
                       samples: int) -> list[float]:
    """Stratified paired bootstrap, preserving 50 observations per lineup."""
    draws = []
    for _ in range(samples):
        sampled = []
        for lineup in LINEUPS:
            values = differences[lineup]
            sampled.extend(values[rng.randrange(len(values))] for _ in values)
        draws.append(statistics.fmean(sampled))
    draws.sort()
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_918)
    args = parser.parse_args()

    all_rows = {"0000": [], "0200": []}
    score_differences = {}
    lineups = {}
    initial_hashes_match = True
    for lineup in LINEUPS:
        baseline, baseline_initial = read_rounds(
            args.results / f"validation_holdout_0000_{lineup}.json")
        challenger, challenger_initial = read_rounds(
            args.results / f"validation_holdout_0200_{lineup}.json")
        hashes_match = baseline_initial == challenger_initial
        initial_hashes_match &= hashes_match
        if not hashes_match:
            raise ValueError(f"Initial conditions differ for {lineup}")
        differences = [new["score"] - old["score"]
                       for old, new in zip(baseline, challenger)]
        score_differences[lineup] = differences
        all_rows["0000"].extend(baseline)
        all_rows["0200"].extend(challenger)
        lineups[lineup] = {
            "rounds": len(baseline),
            "mean_score_0000": statistics.fmean(row["score"] for row in baseline),
            "mean_score_0200": statistics.fmean(row["score"] for row in challenger),
            "paired_score_difference": statistics.fmean(differences),
            "0200_higher_equal_lower": {
                "higher": sum(value > 0 for value in differences),
                "equal": sum(value == 0 for value in differences),
                "lower": sum(value < 0 for value in differences),
            },
            "initial_conditions_match": hashes_match,
        }

    metrics = {tag: mean_metrics(rows) for tag, rows in all_rows.items()}
    deltas = {metric: metrics["0200"][metric] - metrics["0000"][metric]
              for metric in ROUND_METRICS}
    pooled_differences = [value for lineup in LINEUPS
                          for value in score_differences[lineup]]
    payload = {
        "protocol": {
            "paired_initial_conditions": len(pooled_differences),
            "rounds_per_lineup": len(score_differences[LINEUPS[0]]),
            "lineups": list(LINEUPS),
            "initial_condition_hashes_match": initial_hashes_match,
            "bootstrap": {
                "method": "stratified paired percentile bootstrap",
                "samples": args.bootstrap_samples,
                "seed": args.bootstrap_seed,
            },
        },
        "lineups": lineups,
        "pooled": {
            "0000": metrics["0000"],
            "0200": metrics["0200"],
            "0200_minus_0000": deltas,
            "score_difference_95_interval": bootstrap_interval(
                score_differences, random.Random(args.bootstrap_seed),
                args.bootstrap_samples),
            "0200_higher_equal_lower": {
                "higher": sum(value > 0 for value in pooled_differences),
                "equal": sum(value == 0 for value in pooled_differences),
                "lower": sum(value < 0 for value in pooled_differences),
            },
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
