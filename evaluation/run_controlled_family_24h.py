"""Resumable equal-budget training and capability evaluation for five agents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments" / "configs" / "controlled_family_24h.json"
RESULT_ROOT = ROOT / "results" / "controlled_family_24h"
CHECKPOINT_ROOT = RESULT_ROOT / "checkpoints"
RAW_ROOT = RESULT_ROOT / "raw"
LOG_ROOT = RESULT_ROOT / "logs"

MODEL_FILES = {
    "controlled_linear_q": "model.npz",
    "controlled_sarsa_lambda": "model.npz",
    "controlled_ppo": "model.pt",
    "controlled_dddqn_nod4": "model.pt",
    "controlled_dddqn_d4": "model.pt",
}

EVAL_SUITES = {
    "task1_coin": ("coin-heaven", (), 81001),
    "task2_solo": ("classic", (), 82001),
    "task3_peaceful": ("classic", ("peaceful_agent",) * 3, 83001),
    "task4_rule": ("classic", ("rule_based_agent",) * 3, 84001),
    "task4_mixed": (
        "classic",
        ("rule_based_agent", "coin_collector_agent", "peaceful_agent"),
        85001,
    ),
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def live_model(agent: str) -> Path:
    return ROOT / "agent_code" / agent / MODEL_FILES[agent]


def checkpoint(agent: str, training_seed: int, label: str) -> Path:
    return CHECKPOINT_ROOT / f"seed_{training_seed}" / agent / label / MODEL_FILES[agent]


def valid_stats(path: Path, agent: str, rounds: int) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [name for name in payload.get("by_agent", {}) if name == agent]
    return (
        len(payload.get("by_round", {})) == rounds
        and matches == [agent]
        and int(payload["by_agent"][agent].get("rounds", 0)) == rounds
    )


def d4_value(agent: str) -> str:
    return "1" if agent == "controlled_dddqn_d4" else "0"


def run_command(command: list[str], env: dict, log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{tail}")
    return elapsed


def command(agent: str, scenario: str, opponents: tuple[str, ...], rounds: int,
            seed: int, output: Path, train: bool) -> list[str]:
    return [
        sys.executable,
        "main.py",
        "play",
        "--no-gui",
        "--agents",
        agent,
        *opponents,
        "--train",
        "1" if train else "0",
        "--scenario",
        scenario,
        "--n-rounds",
        str(rounds),
        "--seed",
        str(seed),
        "--save-stats",
        str(output),
    ]


def controlled_env(agent: str, agent_seed: int) -> dict:
    env = os.environ.copy()
    env.update({
        "CONTROLLED_AGENT_SEED": str(agent_seed),
        "CONTROLLED_D4": d4_value(agent),
        "PYTHONHASHSEED": str(agent_seed),
    })
    return env


def archive_live(agent: str, destination: Path) -> str:
    source = live_model(agent)
    if not source.exists():
        raise FileNotFoundError(f"Training did not create {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return sha256(destination)


def restore(source: Path, agent: str) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copy2(source, live_model(agent))


def train_stage(agent: str, training_seed: int, label: str, scenario: str,
                opponents: tuple[str, ...], rounds: int, env_seed: int,
                predecessor: Path | None) -> Path:
    destination = checkpoint(agent, training_seed, label)
    stats = RAW_ROOT / "training" / f"seed_{training_seed}" / agent / f"{label}.json"
    record = stats.with_suffix(".run.json")
    if destination.exists() and valid_stats(stats, agent, rounds) and record.exists():
        restore(destination, agent)
        print(f"REUSE train seed={training_seed} agent={agent} stage={label}", flush=True)
        return destination

    if predecessor is None:
        current = live_model(agent)
        if current.exists():
            current.unlink()
        input_hash = None
    else:
        restore(predecessor, agent)
        input_hash = sha256(predecessor)

    stats.parent.mkdir(parents=True, exist_ok=True)
    cmd = command(agent, scenario, opponents, rounds, env_seed, stats, train=True)
    log = LOG_ROOT / "training" / f"seed_{training_seed}" / agent / f"{label}.log"
    elapsed = run_command(cmd, controlled_env(agent, env_seed), log)
    if not valid_stats(stats, agent, rounds):
        raise ValueError(f"Invalid training statistics: {stats}")
    output_hash = archive_live(agent, destination)
    atomic_json(record, {
        "agent": agent,
        "training_seed": training_seed,
        "stage": label,
        "scenario": scenario,
        "opponents": list(opponents),
        "rounds": rounds,
        "environment_seed": env_seed,
        "agent_seed": env_seed,
        "d4": bool(int(d4_value(agent))),
        "input_checkpoint_sha256": input_hash,
        "output_checkpoint": str(destination.relative_to(ROOT)),
        "output_checkpoint_sha256": output_hash,
        "elapsed_seconds": elapsed,
        "command": cmd[1:],
    })
    print(
        f"DONE train seed={training_seed} agent={agent} stage={label} "
        f"rounds={rounds} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return destination


def train_agent(config: dict, agent: str, training_seed: int) -> dict[str, Path]:
    previous = None
    for index, stage in enumerate(config["phase_a"], start=1):
        previous = train_stage(
            agent,
            training_seed,
            stage["name"],
            stage["scenario"],
            tuple(stage["opponents"]),
            int(stage["rounds"]),
            training_seed + index,
            previous,
        )
    anchor = previous
    checkpoints = {"anchor": anchor}
    for branch_index, budget in enumerate(config["task4_branches"], start=1):
        label = f"task4_{int(budget):03d}"
        checkpoints[label] = train_stage(
            agent,
            training_seed,
            label,
            "classic",
            tuple(config["task4_training_opponents"]),
            int(budget),
            training_seed + 100 + branch_index,
            anchor,
        )
    return checkpoints


def evaluate(agent: str, training_seed: int, label: str, checkpoint_path: Path,
             suite: str, rounds: int) -> Path:
    scenario, opponents, base_seed = EVAL_SUITES[suite]
    output = RAW_ROOT / "evaluation" / f"seed_{training_seed}" / agent / label / f"{suite}.json"
    record = output.with_suffix(".run.json")
    if valid_stats(output, agent, rounds) and record.exists():
        print(f"REUSE eval seed={training_seed} agent={agent} ckpt={label} suite={suite}", flush=True)
        return output
    restore(checkpoint_path, agent)
    before_hash = sha256(live_model(agent))
    eval_seed = base_seed
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = command(agent, scenario, opponents, rounds, eval_seed, output, train=False)
    log = LOG_ROOT / "evaluation" / f"seed_{training_seed}" / agent / label / f"{suite}.log"
    elapsed = run_command(cmd, controlled_env(agent, eval_seed), log)
    if not valid_stats(output, agent, rounds):
        raise ValueError(f"Invalid evaluation statistics: {output}")
    after_hash = sha256(live_model(agent))
    if after_hash != before_hash:
        raise RuntimeError(f"Evaluation modified checkpoint for {agent} {label}")
    atomic_json(record, {
        "agent": agent,
        "training_seed": training_seed,
        "checkpoint": label,
        "checkpoint_sha256": before_hash,
        "suite": suite,
        "scenario": scenario,
        "opponents": list(opponents),
        "rounds": rounds,
        "evaluation_seed": eval_seed,
        "elapsed_seconds": elapsed,
        "command": cmd[1:],
    })
    print(
        f"DONE eval seed={training_seed} agent={agent} ckpt={label} suite={suite} "
        f"rounds={rounds} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return output


def evaluate_agent(config: dict, agent: str, training_seed: int,
                   checkpoints: dict[str, Path], tier: int) -> None:
    curve_rounds = int(config["evaluation"]["task4_curve_rounds"])
    capability_rounds = int(config["evaluation"]["capability_rounds"])
    for label in ("anchor", "task4_025", "task4_050", "task4_100"):
        evaluate(agent, training_seed, label, checkpoints[label], "task4_rule", curve_rounds)
    for label in ("anchor", "task4_100"):
        for suite in ("task1_coin", "task2_solo", "task3_peaceful"):
            evaluate(agent, training_seed, label, checkpoints[label], suite, capability_rounds)
    if tier >= 3:
        evaluate(
            agent,
            training_seed,
            "task4_100",
            checkpoints["task4_100"],
            "task4_mixed",
            curve_rounds,
        )


def run(tier: int) -> None:
    config = load_config()
    selected_seeds = config["training_seeds"][: int(config["tiers"][str(tier)])]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(RESULT_ROOT / "active_protocol.json", {**config, "active_tier": tier})
    started = time.monotonic()
    completed = []
    for training_seed in selected_seeds:
        for agent in config["models"]:
            checkpoints = train_agent(config, agent, int(training_seed))
            evaluate_agent(config, agent, int(training_seed), checkpoints, tier)
            completed.append({"training_seed": training_seed, "agent": agent})
            atomic_json(RESULT_ROOT / "progress.json", {
                "tier": tier,
                "completed": completed,
                "elapsed_seconds": time.monotonic() - started,
            })
    atomic_json(RESULT_ROOT / f"tier{tier}_complete.json", {
        "tier": tier,
        "completed": completed,
        "elapsed_seconds": time.monotonic() - started,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    run(args.tier)


if __name__ == "__main__":
    main()
