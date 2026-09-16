"""Isolated, reproducible P2/P3 ablations and a bounded 200-round continuation."""
import argparse
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FLAGS = {"baseline": (False, False), "p2": (True, False), "p3": (False, True), "both": (True, True)}
EVAL_LINEUPS = {
    "rule3": ["rule_based_agent"] * 3,
    "mixed": ["feature_is_everything", "deep_learning_killer", "rule_based_agent"],
}
TRAIN_LINEUPS = [
    ["rule_based_agent"] * 3,
    ["rule_based_agent"] * 3,
    ["rule_based_agent", "dueling_engine_aligned", "peaceful_agent"],
    ["rule_based_agent", "dueling_engine_aligned", "rule_based_agent"],
] * 2


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(arena, senior):
    if arena.exists():
        raise FileExistsError(arena)
    arena.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "logs", "*.log")
    for path in ROOT.glob("*.py"):
        shutil.copy2(path, arena / path.name)
    for folder in ("assets", "evaluation"):
        shutil.copytree(ROOT / folder, arena / folder, ignore=ignore)
    for name in ("dueling_priority23", "dueling_engine_aligned", "rule_based_agent", "peaceful_agent"):
        shutil.copytree(ROOT / "agent_code" / name, arena / "agent_code" / name, ignore=ignore)
    for name in ("feature_is_everything", "deep_learning_killer"):
        shutil.copytree(senior / "agent_code" / name, arena / "agent_code" / name, ignore=ignore)
    for folder in ("logs", "results", "replays", "checkpoints"):
        (arena / folder).mkdir()
    protocol = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "senior_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=senior, text=True).strip(),
        "flags": FLAGS, "training_rounds": 200, "training_lineups_25_rounds_each": TRAIN_LINEUPS,
        "training_seed": 917000, "validation_seed": 918000, "holdout_seed": 919000,
        "validation_rounds_per_lineup": 10, "holdout_rounds_per_lineup": 20,
        "checkpoint_every": 100, "learning_rate": 0.0003,
        "fresh_replay_and_optimizer": True, "new_features_zero_input_weights": True,
        "unchanged": "reward constants, objective Q bias 1.0, center Q bias 0.04, architecture hidden layers",
        "selection": "validation equal-weight mean score across two lineups; include baseline and all frozen/100/200 candidates",
        "promotion": "not automatic; selected candidate must be compared to baseline on fresh holdout, with suicide/invalid regressions disclosed",
        "source_hashes": {str(p.relative_to(arena)): sha(p) for p in arena.rglob("*.py")},
        "initial_weight_sha256": sha(arena / "agent_code/dueling_priority23/dueling_ddqn_model.pt"),
    }
    (arena / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("PREPARED", arena, flush=True)


def run(args):
    arena = args.arena.resolve()
    os.chdir(arena)
    sys.path.insert(0, str(arena))
    import numpy as np
    import torch
    import main
    from environment import BombeRLeWorld
    from evaluation.engine_timing_ablation import CPUModelUnpickler
    from evaluation.summarize_stats import summarize_match
    torch.set_num_threads(1)
    torch.manual_seed(917000)
    training = args.mode == "train"
    tag = "train" if training else args.tag
    name = "priority23_learner" if training else "priority23_eval_" + tag
    destination = arena / "agent_code" / name
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(arena / "agent_code/dueling_priority23", destination,
                    ignore=shutil.ignore_patterns("__pycache__", "logs"))
    if args.checkpoint:
        shutil.copy2(args.checkpoint.resolve(), destination / "dueling_ddqn_model.pt")
    flags = (True, True) if training else FLAGS[args.arm]
    callbacks = importlib.import_module(f"agent_code.{name}.callbacks")
    original_setup = callbacks.setup
    learner = {}

    def setup(agent):
        if training and "runtime" in learner:
            logger = agent.logger
            agent.__dict__.update(learner["runtime"].__dict__)
            agent.logger = logger
        else:
            # Fixed CPU runtime for this measured experiment, including training.
            train_flag = agent.train
            agent.train = False
            original_setup(agent)
            agent.train = train_flag
        agent.feature_extractor.enable_economy, agent.feature_extractor.enable_contest = flags
        if training:
            learner["runtime"] = agent
    callbacks.setup = setup
    if training:
        train = importlib.import_module(f"agent_code.{name}.train")
        original_training_setup = train.setup_training
        def setup_training(agent):
            if not hasattr(agent, "replay"):
                agent.loaded_optimizer_state = None
                original_training_setup(agent)
        train.setup_training = setup_training

    senior = importlib.import_module("agent_code.feature_is_everything.callbacks")
    def cpu_senior(agent):
        agent.device = torch.device("cpu")
        agent.feature = senior.Feature()
        with open("my-saved-model.pt", "rb") as handle:
            agent.policy_net = CPUModelUnpickler(handle).load().to(agent.device)
        agent.policy_net.eval()
    senior.setup = cpu_senior

    current = {"label": tag, "initial": [], "offset": 0}
    log_root = arena / "logs" / tag
    log_root.mkdir(exist_ok=True)
    original_handler = logging.FileHandler
    logging.FileHandler = lambda filename, *a, **kw: original_handler(
        log_root / (current["label"] + "_" + Path(filename).name), *a, **kw)
    setup_agents = BombeRLeWorld.setup_agents
    new_round = BombeRLeWorld.new_round
    end_round = BombeRLeWorld.end_round
    started = time.monotonic()

    def seeded_setup(world, agents):
        setup_agents(world, agents)
        world.agents[0].name = "candidate"
        world.agents[0].display_name = "candidate"

    def seeded_round(world):
        seed = current["base"] + world.round
        world.rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        for i, agent in enumerate(world.agents):
            runtime = agent.backend.runner.fake_self
            if hasattr(runtime, "rng"):
                runtime.rng.seed(seed + i)
            if hasattr(runtime, "replay_rng"):
                runtime.replay_rng = np.random.default_rng(seed + i)
        new_round(world)
        initial = {"field": world.arena.tolist(), "coins": [list(map(int, c.get_state())) for c in world.coins],
                   "positions": [[int(a.x), int(a.y)] for a in world.agents]}
        current["initial"].append({"seed": seed, "sha256": hashlib.sha256(
            json.dumps(initial, sort_keys=True).encode()).hexdigest()})

    def record_round(world):
        end_round(world)
        count = current["offset"] + world.round
        row = world.round_statistics[world.round_id]["agents"]["candidate"]
        print(f"{tag} {current['label']} round={count} score={row['score']} "
              f"suicide={row['suicides']} elapsed={time.monotonic()-started:.1f}s", flush=True)
        if training and count in (100, 200):
            cp = arena / "checkpoints" / f"trained_{count:04d}.pt"
            shutil.copy2(destination / "dueling_ddqn_model.pt", cp)
            agent = learner["runtime"]
            print("CHECKPOINT", str(cp), agent.environment_steps, agent.learner_updates, flush=True)
    BombeRLeWorld.setup_agents = seeded_setup
    BombeRLeWorld.new_round = seeded_round
    BombeRLeWorld.end_round = record_round

    summaries = []
    schedules = [(f"block_{i+1}", opp, 917000 + 25*i, min(25, args.rounds - 25*i), 25*i)
                 for i, opp in enumerate(TRAIN_LINEUPS) if 25*i < args.rounds] if training else [
        (label, opponents, args.seed + 100*i, args.rounds, 0)
        for i, (label, opponents) in enumerate(EVAL_LINEUPS.items())]
    for label, opponents, seed, rounds, offset in schedules:
        current.update(label=label, base=seed, offset=offset, initial=[])
        output = arena / "results" / f"{tag}_{label}.json"
        if output.exists():
            raise FileExistsError(output)
        argv = ["play", "--no-gui", "--agents", name, *opponents,
                "--scenario", "classic", "--n-rounds", str(rounds), "--seed", str(seed),
                "--save-stats", str(output)]
        if training:
            argv += ["--train", "1"]
        main.main(argv)
        payload = json.loads(output.read_text())
        payload["evaluation"] = {"training": training, "flags": flags, "initial_conditions": current["initial"],
                                 "checkpoint": str(args.checkpoint) if args.checkpoint else "unchanged original weights"}
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        row = next(r for r in summarize_match([output]) if r["agent"] == "candidate")
        summaries.append({"lineup": label, **row})
        print("RESULT", tag, label, json.dumps(row), flush=True)
        for logger in logging.Logger.manager.loggerDict.values():
            if isinstance(logger, logging.Logger):
                for handler in list(logger.handlers):
                    handler.close()
                    logger.removeHandler(handler)
    (arena / "results" / f"{tag}_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "train", "eval"])
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--senior-root", type=Path)
    parser.add_argument("--arm", choices=FLAGS, default="both")
    parser.add_argument("--tag", default="both")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=918000)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if not args.senior_root:
            parser.error("--senior-root is required")
        prepare(args.arena.resolve(), args.senior_root.resolve())
    else:
        if args.rounds < 1 or (args.mode == "train" and args.rounds > 200):
            parser.error("Training is bounded to 1..200 rounds")
        if args.checkpoint:
            args.checkpoint = args.checkpoint.resolve()
        run(args)
