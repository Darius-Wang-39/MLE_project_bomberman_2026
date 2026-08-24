"""Run a reproducible multi-lineup evaluation suite and aggregate its stats."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.summarize_stats import summarize


def build_command(candidate: str, lineup: dict, output: Path):
    opponents = list(lineup["opponents"])
    if len(opponents) != 3:
        raise ValueError("Each Bomberman evaluation lineup requires exactly three opponents")
    command = [
        sys.executable,
        "main.py",
        "play",
        "--no-gui",
        "--agents",
        candidate,
        *opponents,
        "--scenario",
        lineup.get("scenario", "classic"),
        "--n-rounds",
        str(int(lineup["rounds"])),
        "--seed",
        str(int(lineup["seed"])),
        "--save-stats",
        str(output),
    ]
    if lineup.get("silence_errors", False):
        command.append("--silence-errors")
    return command


def run_suite(config_path: Path, report_path: Path, resume: bool = False):
    config_path = config_path.resolve()
    report_path = report_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidate = config["candidate"]
    raw_directory = report_path.parent / (report_path.stem + "_raw")
    raw_directory.mkdir(parents=True, exist_ok=True)
    reports = []
    raw_paths = []
    for lineup in config["lineups"]:
        raw_path = raw_directory / f"{lineup['name']}.json"
        command = build_command(candidate, lineup, raw_path)
        if not (resume and raw_path.exists()):
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        metrics = summarize([raw_path], candidate)
        reports.append(
            {
                **lineup,
                "raw_stats": str(raw_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "command": command[1:],
                "metrics": metrics,
            }
        )
        raw_paths.append(raw_path)
    payload = {
        "suite": config["suite"],
        "candidate": candidate,
        "lineups": reports,
        "aggregate": summarize(raw_paths, candidate),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_suite(args.config, args.report, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
