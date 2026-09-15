"""Summarize one or more JSON files produced by main.py --save-stats."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


SUM_FIELDS = ("rounds", "steps", "score", "coins", "crates", "bombs", "kills", "suicides", "invalid", "time")
ROUND_FIELDS = ("score", "coins", "crates", "bombs", "kills", "suicides", "invalid", "steps", "time")


def summarize(paths, agent_name: str):
    totals = defaultdict(float)
    matched_files = 0
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        agents = payload.get("by_agent", {})
        candidates = [name for name in agents if name == agent_name or name.startswith(agent_name + "_")]
        if not candidates:
            continue
        matched_files += 1
        for name in candidates:
            for field in SUM_FIELDS:
                totals[field] += float(agents[name].get(field, 0.0))

    rounds = totals["rounds"]
    steps = totals["steps"]
    return {
        "agent": agent_name,
        "files": matched_files,
        "rounds": int(rounds),
        "mean_score": totals["score"] / rounds if rounds else 0.0,
        "coins_per_round": totals["coins"] / rounds if rounds else 0.0,
        "crates_per_round": totals["crates"] / rounds if rounds else 0.0,
        "kills_per_round": totals["kills"] / rounds if rounds else 0.0,
        "suicides_per_round": totals["suicides"] / rounds if rounds else 0.0,
        "invalid_per_round": totals["invalid"] / rounds if rounds else 0.0,
        "mean_action_ms": 1000.0 * totals["time"] / steps if steps else 0.0,
        "total_steps": int(steps),
    }


def summarize_match(paths):
    """Summarize all agents from per-round tournament records.

    A tied highest score counts as a win for every tied agent. Rank is the
    competition rank (one plus the number of strictly higher-scoring agents).
    Files produced before per-agent round records were added are skipped.
    """
    by_agent = defaultdict(list)
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for round_data in payload.get("by_round", {}).values():
            agents = round_data.get("agents", {})
            if not agents:
                continue
            scores = {name: float(values.get("score", 0.0)) for name, values in agents.items()}
            for name, values in agents.items():
                opponents = [score for other, score in scores.items() if other != name]
                score = scores[name]
                by_agent[name].append({
                    **{field: float(values.get(field, 0.0)) for field in ROUND_FIELDS},
                    "survived": bool(values.get("survived", False)),
                    "is_match": bool(opponents),
                    "win": score == max(scores.values()) if opponents else None,
                    "rank": 1 + sum(other_score > score for other_score in scores.values()) if opponents else None,
                    "score_margin": score - sum(opponents) / len(opponents) if opponents else None,
                })

    result = []
    for name, rounds in sorted(by_agent.items()):
        count = len(rounds)
        totals = {field: sum(row[field] for row in rounds) for field in ROUND_FIELDS}
        match_rounds = [row for row in rounds if row["is_match"]]
        result.append({
            "agent": name,
            "rounds": count,
            "mean_score": totals["score"] / count,
            "win_rate": sum(row["win"] for row in match_rounds) / len(match_rounds) if match_rounds else float("nan"),
            "mean_rank": sum(row["rank"] for row in match_rounds) / len(match_rounds) if match_rounds else float("nan"),
            "mean_score_margin": sum(row["score_margin"] for row in match_rounds) / len(match_rounds) if match_rounds else float("nan"),
            "coins_per_round": totals["coins"] / count,
            "crates_per_round": totals["crates"] / count,
            "bombs_per_round": totals["bombs"] / count,
            "kills_per_round": totals["kills"] / count,
            "suicides_per_round": totals["suicides"] / count,
            "invalid_per_round": totals["invalid"] / count,
            "survival_rate": sum(row["survived"] for row in rounds) / count,
            "mean_action_ms": 1000.0 * totals["time"] / totals["steps"] if totals["steps"] else 0.0,
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--agent", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.paths, args.agent), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
