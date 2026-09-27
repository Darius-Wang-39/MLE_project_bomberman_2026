"""Run the E3 economic-routing robustness extension on two added lineups.

This module reuses the locked E3 runner and estimands while keeping a separate
protocol and result root, so the completed main experiment remains immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import e3_economic_routing as base


EXTENSION_ROOT = ROOT / "results" / "e3_economic_routing_extension"
LINEUPS = {
    "rule3": ["rule_based_agent"] * 3,
    "internal_mixed": [
        "frozen_ddqn_day7", "frozen_sarsa_best", "frozen_linear_q_final",
    ],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_base() -> None:
    base.LINEUPS = LINEUPS
    base.RESULT_ROOT = EXTENSION_ROOT
    base.PROTOCOL_PATH = EXTENSION_ROOT / "protocol.json"
    base.SUMMARY_PATH = EXTENSION_ROOT / "summary.json"


def extension_protocol() -> dict:
    configure_base()
    spec = base.protocol()
    spec.update({
        "experiment": "E3 economic-routing added-lineup robustness extension",
        "status": "post-main-experiment requested extension",
        "relationship_to_main": (
            "same candidate, variants, seeds, rounds, instrumentation, estimands, "
            "and inference; only the two opponent lineups are added"
        ),
        "main_protocol": "results/e3_economic_routing/protocol.json",
        "main_protocol_sha256": digest(
            ROOT / "results" / "e3_economic_routing" / "protocol.json"
        ),
        "lineups": LINEUPS,
        "total_rounds": len(base.VARIANTS) * len(LINEUPS) * len(base.SEEDS) * base.ROUNDS_PER_SEED,
        "extension_wrapper_sha256": digest(Path(__file__)),
    })
    return spec


def main_cli() -> None:
    configure_base()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--variant", choices=base.VARIANTS, required=True)
    run_parser.add_argument("--lineup", choices=LINEUPS, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    sub.add_parser("summarize")
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "prepare":
        spec = extension_protocol()
        if base.PROTOCOL_PATH.exists():
            existing = json.loads(base.PROTOCOL_PATH.read_text(encoding="utf-8"))
            if existing != spec:
                raise ValueError("E3 extension protocol already locked and differs")
        base.atomic_json(base.PROTOCOL_PATH, spec)
    elif args.command == "run":
        if args.seed not in base.SEEDS:
            raise ValueError("Seed outside locked E3 set")
        base.run_one(args.variant, args.lineup, args.seed)
    elif args.command == "summarize":
        base.summarize()
    else:
        print(json.dumps(base.status(), indent=2))


if __name__ == "__main__":
    main_cli()
