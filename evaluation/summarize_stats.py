"""Summarize one or more JSON files produced by main.py --save-stats."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


SUM_FIELDS = ("rounds", "steps", "score", "coins", "crates", "bombs", "kills", "suicides", "invalid", "time")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--agent", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.paths, args.agent), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

