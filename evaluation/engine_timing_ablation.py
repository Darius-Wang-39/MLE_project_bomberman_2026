"""Frozen-weight 2x2 timing/objective-bias evaluation in an isolated arena.

Prepare once with --prepare --arena PATH --senior-root CHECKOUT; then run
--arena PATH --arm baseline|timing|no_bias|both (one process per arm is safe).
Each arm plays 20 predeclared seeds per lineup. No training or promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import logging
import os
import pickle
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "baseline": ("dueling_hunt_center_04", 1.0),
    "timing": ("dueling_engine_aligned", 1.0),
    "no_bias": ("dueling_hunt_center_04", 0.0),
    "both": ("dueling_engine_aligned", 0.0),
}
LINEUPS = {
    "rule3": (916201, ["rule_based_agent"] * 3),
    "mixed": (916301, ["feature_is_everything", "deep_learning_killer", "rule_based_agent"]),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(arena, senior):
    if arena.exists():
        raise FileExistsError(f"Refusing to overwrite arena: {arena}")
    if not (senior / "agent_code/feature_is_everything/my-saved-model.pt").is_file():
        raise FileNotFoundError("Expected the previous-cohort checkout with frozen weights")
    arena.mkdir(parents=True)
    for source in ROOT.glob("*.py"):
        shutil.copy2(source, arena / source.name)
    shutil.copytree(ROOT / "assets", arena / "assets")
    names = {value[0] for value in ARMS.values()} | {"rule_based_agent"}
    ignore = shutil.ignore_patterns("__pycache__", "logs", "*.log")
    for name in sorted(names):
        shutil.copytree(ROOT / "agent_code" / name, arena / "agent_code" / name, ignore=ignore)
    for name in ("feature_is_everything", "deep_learning_killer"):
        shutil.copytree(senior / "agent_code" / name, arena / "agent_code" / name, ignore=ignore)
    shutil.copytree(ROOT / "evaluation", arena / "evaluation", ignore=ignore)
    for name in ("logs", "results", "replays"):
        (arena / name).mkdir()
    protocol = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "senior_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=senior, text=True).strip(),
        "arms": ARMS, "lineups": LINEUPS, "rounds_per_lineup": 20,
        "training": False, "center_bias": 0.04,
        "seeds": "base seed + zero-based round index; reset environment and policy RNG each round",
        "pairing": "same initial map, hidden coins, player positions and candidate display name",
        "primary_metric": "mean game score, reported separately for both lineups",
        "secondary_metrics": "tied-first rate, suicides, invalid actions, coins, bombs, kills",
        "replay_selection": "last predeclared round of each lineup, never chosen for a favorable result",
        "weight_hashes": {str(p.relative_to(arena)): digest(p) for p in (arena / "agent_code").rglob("*")
                          if p.suffix in (".pt", ".json")},
        "source_hashes": {str(p.relative_to(arena)): digest(p) for p in arena.rglob("*.py")},
    }
    (arena / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Prepared {arena}", flush=True)


class CPUModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        import torch
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return lambda data: torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        allowed = {
            ("collections", "OrderedDict"), ("builtins", "set"),
            ("torch._utils", "_rebuild_parameter"), ("torch._utils", "_rebuild_tensor_v2"),
            ("torch.nn.modules.container", "Sequential"), ("torch.nn.modules.linear", "Linear"),
            ("torch.nn.modules.normalization", "LayerNorm"), ("torch.nn.modules.activation", "ReLU"),
            ("agent_code.feature_is_everything.model", "DQN"),
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unexpected checkpoint global: {module}.{name}")
        return super().find_class(module, name)


def run(arena, arm, rounds):
    sys.path.insert(0, str(arena))
    os.chdir(arena)
    import numpy as np
    import torch
    import main
    from environment import BombeRLeWorld
    from evaluation.summarize_stats import summarize_match

    protocol = json.loads((arena / "protocol.json").read_text(encoding="utf-8"))
    for path, expected in {**protocol["source_hashes"], **protocol["weight_hashes"]}.items():
        if digest(arena / path) != expected:
            raise ValueError(f"Arena changed since protocol was locked: {path}")
    torch.set_num_threads(1)
    torch.manual_seed(916000)
    name, bias = ARMS[arm]
    callbacks = importlib.import_module(f"agent_code.{name}.callbacks")
    callbacks.OBJECTIVE_Q_BIAS = bias
    senior = importlib.import_module("agent_code.feature_is_everything.callbacks")

    def cpu_setup(agent):
        agent.device = torch.device("cpu")
        agent.feature = senior.Feature()
        with open("my-saved-model.pt", "rb") as handle:
            agent.policy_net = CPUModelUnpickler(handle).load().to(agent.device)
        agent.policy_net.eval()
    senior.setup = cpu_setup

    log_root = arena / "logs" / arm
    log_root.mkdir(exist_ok=True)
    original_handler = logging.FileHandler
    logging.FileHandler = lambda filename, *a, **kw: original_handler(log_root / Path(filename).name, *a, **kw)
    original_setup = BombeRLeWorld.setup_agents
    original_new = BombeRLeWorld.new_round
    original_end = BombeRLeWorld.end_round
    current = {}

    def setup_agents(world, agents):
        original_setup(world, agents)
        # Keep the name exposed to opponent features constant across all arms.
        world.agents[0].name = "candidate"
        world.agents[0].display_name = "candidate"

    def new_round(world):
        seed = current["base"] + world.round
        world.rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        for index, agent in enumerate(world.agents):
            runtime = agent.backend.runner.fake_self
            if hasattr(runtime, "rng"):
                runtime.rng.seed(seed + index)
        original_new(world)
        initial = {"arena": world.arena.tolist(),
                   "coins": [list(map(int, c.get_state())) for c in world.coins],
                   "positions": [[int(a.x), int(a.y)] for a in world.agents]}
        current["initial"].append({"seed": seed, "sha256": hashlib.sha256(
            json.dumps(initial, sort_keys=True).encode()).hexdigest()})

    def end_round(world):
        original_end(world)
        row = world.round_statistics[world.round_id]["agents"]["candidate"]
        print(f"{arm} {current['lineup']} {world.round}/{rounds} "
              f"score={row['score']} suicides={row['suicides']} invalid={row['invalid']}", flush=True)

    BombeRLeWorld.setup_agents = setup_agents
    BombeRLeWorld.new_round = new_round
    BombeRLeWorld.end_round = end_round
    summaries = []
    for lineup, (base, opponents) in LINEUPS.items():
        current.update(base=base, lineup=lineup, initial=[])
        output = arena / "results" / f"{arm}_{lineup}.json"
        if output.exists():
            raise FileExistsError(output)
        main.main(["play", "--no-gui", "--agents", name, *opponents,
                   "--scenario", "classic", "--n-rounds", str(rounds), "--seed", str(base),
                   "--save-stats", str(output), "--match-name", f"{arm}_{lineup}",
                   "--save-replay", str(arena / "replays" / f"{arm}_{lineup}_last.pt")])
        payload = json.loads(output.read_text())
        payload["evaluation"] = {"arm": arm, "agent": name, "objective_q_bias": bias,
                                 "training": False, "initial_conditions": current["initial"],
                                 "lineup": [name, *opponents]}
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary = next(row for row in summarize_match([output]) if row["agent"] == "candidate")
        summaries.append({"lineup": lineup, **summary})
        print("RESULT", arm, lineup, json.dumps(summary), flush=True)
        for logger in logging.Logger.manager.loggerDict.values():
            if isinstance(logger, logging.Logger):
                for handler in list(logger.handlers):
                    handler.close()
                    logger.removeHandler(handler)
    for path, expected in protocol["weight_hashes"].items():
        assert digest(arena / path) == expected, f"Weights changed: {path}"
    (arena / "results" / f"{arm}_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--senior-root", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--rounds", type=int, default=20, help="Use 1 for a separate smoke arena")
    args = parser.parse_args()
    if args.prepare:
        if args.senior_root is None:
            parser.error("--prepare needs --senior-root")
        prepare(args.arena.resolve(), args.senior_root.resolve())
    else:
        if args.arm is None or args.rounds < 1:
            parser.error("Supply --arm and positive --rounds")
        run(args.arena.resolve(), args.arm, args.rounds)
