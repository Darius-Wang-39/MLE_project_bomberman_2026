"""Train and evaluate the engine-aligned DDQN without P2/P3.

The experiment runs inside a copied arena so the adopted inference agent and
submission archive are never modified.  Training keeps one learner runtime
across 25-round opponent blocks, starts from the adopted frozen weights, uses a
fresh optimizer/replay buffer, and saves checkpoints every 100 rounds.
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
import time


ROOT = Path(__file__).resolve().parents[1]
WORKING_AGENT = "engine_aligned_adapt_train"
BASE_AGENT = "dueling_engine_aligned"
MODEL_FILE = "dueling_ddqn_model.pt"
BLOCK_SIZE = 25
TOTAL_ROUNDS = 1000
CHECKPOINT_EVERY = 100
TRAINING_SEED = 1_017_000
VALIDATION_SEED = 1_027_000

# One 200-round cycle, repeated five times.  This is the agreed 25/25/15/20/15
# approximate mixture at block resolution, with the old DDQN used only as a
# minority auxiliary opponent.
TRAIN_LINEUPS = (
    ("rule3", ("rule_based_agent",) * 3),
    ("feature_rule2", ("feature_is_everything", "rule_based_agent", "rule_based_agent")),
    ("killer_rule2", ("deep_learning_killer", "rule_based_agent", "rule_based_agent")),
    ("senior_mixed", ("feature_is_everything", "deep_learning_killer", "rule_based_agent")),
    ("old_ddqn_rule2", ("dueling_hunt_center_04", "rule_based_agent", "rule_based_agent")),
    ("feature_rule2_repeat", ("feature_is_everything", "rule_based_agent", "rule_based_agent")),
    ("old_ddqn_killer", ("dueling_hunt_center_04", "deep_learning_killer", "rule_based_agent")),
    ("senior_mixed_repeat", ("feature_is_everything", "deep_learning_killer", "rule_based_agent")),
)

VALIDATION_LINEUPS = (
    ("rule3", ("rule_based_agent",) * 3),
    ("senior_mixed", ("feature_is_everything", "deep_learning_killer", "rule_based_agent")),
    ("old_feature_rule", ("dueling_hunt_center_04", "feature_is_everything", "rule_based_agent")),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def prepare(arena: Path) -> None:
    if arena.exists():
        raise FileExistsError(f"Refusing to overwrite arena: {arena}")
    for name in (BASE_AGENT, "dueling_hunt_center_04", "rule_based_agent",
                 "external_feature_is_everything", "external_deep_learning_killer"):
        if not (ROOT / "agent_code" / name).is_dir():
            raise FileNotFoundError(f"Missing agent_code/{name}")

    arena.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "logs", "*.log")
    for source in ROOT.glob("*.py"):
        shutil.copy2(source, arena / source.name)
    shutil.copytree(ROOT / "assets", arena / "assets", ignore=ignore)
    shutil.copytree(ROOT / "evaluation", arena / "evaluation", ignore=ignore)
    (arena / "agent_code").mkdir()
    for name in (BASE_AGENT, "dueling_hunt_center_04", "rule_based_agent"):
        shutil.copytree(ROOT / "agent_code" / name, arena / "agent_code" / name,
                        ignore=ignore)
    for source, destination in (("external_feature_is_everything", "feature_is_everything"),
                                ("external_deep_learning_killer", "deep_learning_killer")):
        shutil.copytree(ROOT / "agent_code" / source, arena / "agent_code" / destination,
                        ignore=ignore)

    working = arena / "agent_code" / WORKING_AGENT
    shutil.copytree(ROOT / "agent_code" / BASE_AGENT, working, ignore=ignore)
    for filename in ("train.py", "replay.py", "symmetry.py"):
        shutil.copy2(ROOT / "agent_code" / "dueling_ddqn_agent" / filename,
                     working / filename)

    for folder in ("logs", "results", "replays", "checkpoints"):
        (arena / folder).mkdir()
    start = arena / "checkpoints" / "engine_aligned_adapt_0000.pt"
    shutil.copy2(working / MODEL_FILE, start)

    schedule = []
    for block_index in range(TOTAL_ROUNDS // BLOCK_SIZE):
        label, opponents = TRAIN_LINEUPS[block_index % len(TRAIN_LINEUPS)]
        schedule.append({
            "block": block_index + 1,
            "cycle": block_index // len(TRAIN_LINEUPS) + 1,
            "round_start": block_index * BLOCK_SIZE + 1,
            "round_end": (block_index + 1) * BLOCK_SIZE,
            "label": label,
            "opponents": list(opponents),
            "seed": TRAINING_SEED + block_index * 100,
        })

    protocol = {
        "experiment": "continue_training_engine_aligned_dddqn",
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "base_agent": BASE_AGENT,
        "working_agent": WORKING_AGENT,
        "p2": False,
        "p3": False,
        "engine_timing_fixes": True,
        "terminal_transition_deduplication": True,
        "total_rounds": TOTAL_ROUNDS,
        "block_size": BLOCK_SIZE,
        "checkpoint_every": CHECKPOINT_EVERY,
        "formal_validation_every": 200,
        "training_schedule": schedule,
        "validation_lineups": {label: list(opponents) for label, opponents in VALIDATION_LINEUPS},
        "validation_rounds_per_lineup": 20,
        "validation_seed": VALIDATION_SEED,
        "initialization": {
            "policy": "adopted engine-aligned frozen policy",
            "optimizer": "fresh AdamW",
            "replay": "fresh prioritized replay; retained across all blocks",
            "target": "fresh copy of initial policy; retained across all blocks",
            "n_step": "fresh; retained/cleared normally by round callbacks",
        },
        "exploration": {
            "counter": "environment_steps minus continuation start environment_steps",
            "segments": [
                {"steps": [0, 65_000], "epsilon": [0.20, 0.12]},
                {"steps": [65_000, 195_000], "epsilon": [0.12, 0.07]},
                {"steps": [195_000, 325_000], "epsilon": [0.07, 0.05]},
                {"steps": [325_000, None], "epsilon": [0.05, 0.05]},
            ],
        },
        "unchanged": [
            "22 action-relative features", "reward constants", "network architecture",
            "learning rate 3e-4", "objective Q bias 1.00", "center Q bias 0.04",
            "five-step returns", "prioritized replay", "D4 augmentation",
        ],
        "checkpoint_zero_sha256": digest(start),
        "source_hashes": {
            str(path.relative_to(arena)): digest(path)
            for path in sorted((arena / "agent_code" / WORKING_AGENT).glob("*.py"))
        },
    }
    write_json(arena / "protocol.json", protocol)
    print(f"PREPARED {arena}", flush=True)


class CPUModelUnpickler(pickle.Unpickler):
    """Load the previous cohort's CUDA-serialized PyTorch model on CPU."""

    def find_class(self, module, name):
        import torch
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return lambda data: torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        allowed = {
            ("collections", "OrderedDict"), ("builtins", "set"),
            ("torch._utils", "_rebuild_parameter"), ("torch._utils", "_rebuild_tensor_v2"),
            ("torch.nn.modules.container", "Sequential"), ("torch.nn.modules.linear", "Linear"),
            ("torch.nn.modules.normalization", "LayerNorm"), ("torch.nn.modules.activation", "ReLU"),
            ("agent_code.external_feature_is_everything.model", "DQN"),
            ("agent_code.feature_is_everything.model", "DQN"),
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unexpected checkpoint global: {module}.{name}")
        return super().find_class(module, name)


def epsilon_for_step(step: int) -> float:
    """Agreed conservative continuation schedule, piecewise linear by action."""
    if step <= 65_000:
        return 0.20 + (0.12 - 0.20) * step / 65_000
    if step <= 195_000:
        return 0.12 + (0.07 - 0.12) * (step - 65_000) / 130_000
    if step <= 325_000:
        return 0.07 + (0.05 - 0.07) * (step - 195_000) / 130_000
    return 0.05


def enter_arena(arena: Path):
    os.chdir(arena)
    sys.path.insert(0, str(arena))
    import numpy as np
    import torch
    import main
    from environment import BombeRLeWorld
    from evaluation.summarize_stats import summarize_match
    torch.set_num_threads(1)
    return np, torch, main, BombeRLeWorld, summarize_match


def patch_senior_cpu(torch) -> None:
    senior = importlib.import_module("agent_code.feature_is_everything.callbacks")

    def cpu_setup(agent):
        agent.device = torch.device("cpu")
        agent.feature = senior.Feature()
        with open("my-saved-model.pt", "rb") as handle:
            agent.policy_net = CPUModelUnpickler(handle).load().to(agent.device)
        agent.policy_net.eval()

    senior.setup = cpu_setup


def close_loggers() -> None:
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)


def train(arena: Path, max_blocks: int | None = None) -> None:
    np, torch, main, world_class, _ = enter_arena(arena)
    protocol = json.loads((arena / "protocol.json").read_text(encoding="utf-8"))
    if any(digest(arena / path) != expected for path, expected in protocol["source_hashes"].items()):
        raise ValueError("Working-agent source differs from locked protocol")
    patch_senior_cpu(torch)
    torch.manual_seed(TRAINING_SEED)

    callbacks = importlib.import_module(f"agent_code.{WORKING_AGENT}.callbacks")
    train_module = importlib.import_module(f"agent_code.{WORKING_AGENT}.train")
    original_callback_setup = callbacks.setup
    original_training_setup = train_module.setup_training
    learner = {}

    def continuation_epsilon(agent):
        if not agent.train:
            return 0.0
        step = max(0, int(agent.environment_steps) - int(agent.continuation_start_environment_steps))
        return epsilon_for_step(step)

    callbacks.exploration_rate = continuation_epsilon

    def callback_setup(agent):
        if "runtime" in learner:
            logger = agent.logger
            agent.__dict__.update(learner["runtime"].__dict__)
            agent.logger = logger
        else:
            train_flag = agent.train
            agent.train = False  # force CPU even on hosts with an accelerator
            original_callback_setup(agent)
            agent.train = train_flag
            agent.continuation_start_environment_steps = int(agent.environment_steps)
        # Every main.main call creates a new backend/fake_self.  Point at that
        # new object after copying the continuous learner state, otherwise the
        # following block would train a stale copy and be discarded.
        learner["runtime"] = agent

    callbacks.setup = callback_setup

    def training_setup(agent):
        if not hasattr(agent, "replay"):
            agent.loaded_optimizer_state = None
            original_training_setup(agent)

    train_module.setup_training = training_setup

    current = {"block": 0, "base": TRAINING_SEED, "initial": []}
    original_setup_agents = world_class.setup_agents
    original_new_round = world_class.new_round
    original_end_round = world_class.end_round
    started = time.monotonic()

    def setup_agents(world, agents):
        original_setup_agents(world, agents)
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
            if hasattr(runtime, "replay_rng"):
                runtime.replay_rng = np.random.default_rng(seed + index)
        original_new_round(world)
        state = {
            "field": world.arena.tolist(),
            "coins": [list(map(int, coin.get_state())) for coin in world.coins],
            "positions": [[int(agent.x), int(agent.y)] for agent in world.agents],
        }
        current["initial"].append({
            "seed": seed,
            "sha256": hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
        })

    def end_round(world):
        original_end_round(world)
        completed = current["offset"] + world.round
        if completed % CHECKPOINT_EVERY == 0:
            source = arena / "agent_code" / WORKING_AGENT / MODEL_FILE
            destination = arena / "checkpoints" / f"engine_aligned_adapt_{completed:04d}.pt"
            shutil.copy2(source, destination)
            checkpoint = torch.load(destination, map_location="cpu", weights_only=True)
            checkpoint["continuation"] = {
                "completed_rounds": completed,
                "base_sha256": protocol["checkpoint_zero_sha256"],
                "protocol_sha256": digest(arena / "protocol.json"),
                "continuation_steps": int(learner["runtime"].environment_steps)
                - int(learner["runtime"].continuation_start_environment_steps),
                "epsilon": continuation_epsilon(learner["runtime"]),
            }
            temporary = destination.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(destination)
            print(f"CHECKPOINT rounds={completed} steps={checkpoint['continuation']['continuation_steps']} "
                  f"epsilon={checkpoint['continuation']['epsilon']:.4f} path={destination}", flush=True)

    world_class.setup_agents = setup_agents
    world_class.new_round = new_round
    world_class.end_round = end_round

    manifest = []
    schedule = protocol["training_schedule"]
    if max_blocks is not None:
        schedule = schedule[:max_blocks]
    for row in schedule:
        block = int(row["block"])
        current.update(block=block, base=int(row["seed"]), offset=int(row["round_start"]) - 1, initial=[])
        output = arena / "results" / f"train_block_{block:02d}.json"
        if output.exists():
            raise FileExistsError(f"Refusing partial/resumed training without runtime state: {output}")
        argv = [
            "play", "--no-gui", "--agents", WORKING_AGENT, *row["opponents"],
            "--scenario", "classic", "--n-rounds", str(BLOCK_SIZE),
            "--seed", str(row["seed"]), "--train", "1", "--save-stats", str(output),
            "--match-name", f"adapt_block_{block:02d}",
        ]
        main.main(argv)
        payload = json.loads(output.read_text(encoding="utf-8"))
        payload["experiment"] = {
            "training": True, "block": block, "cycle": row["cycle"],
            "global_rounds": [row["round_start"], row["round_end"]],
            "opponents": row["opponents"], "seed": row["seed"],
            "initial_conditions": current["initial"],
        }
        write_json(output, payload)
        runtime = learner["runtime"]
        continuation_steps = int(runtime.environment_steps) - int(runtime.continuation_start_environment_steps)
        manifest.append({**row, "stats": str(output.relative_to(arena)),
                         "continuation_steps": continuation_steps,
                         "learner_updates": int(runtime.learner_updates),
                         "epsilon": continuation_epsilon(runtime),
                         "elapsed_seconds": time.monotonic() - started})
        write_json(arena / "training_manifest.json", manifest)
        print(f"BLOCK {block:02d}/40 rounds={row['round_end']} lineup={row['label']} "
              f"steps={continuation_steps} updates={runtime.learner_updates} "
              f"epsilon={continuation_epsilon(runtime):.4f} elapsed={time.monotonic()-started:.1f}s", flush=True)
        close_loggers()
    print(f"TRAINING_COMPLETE elapsed={time.monotonic()-started:.1f}s", flush=True)


def evaluate(arena: Path, checkpoint: Path, tag: str, rounds: int, seed: int,
             diagnose_invalid: bool = False) -> None:
    np, torch, main, world_class, summarize_match = enter_arena(arena)
    patch_senior_cpu(torch)
    candidate = f"engine_aligned_eval_{tag}"
    destination = arena / "agent_code" / candidate
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(arena / "agent_code" / BASE_AGENT, destination,
                    ignore=shutil.ignore_patterns("__pycache__", "logs", "*.log"))
    shutil.copy2(checkpoint, destination / MODEL_FILE)

    diagnostic_rows = []
    if diagnose_invalid:
        callbacks = importlib.import_module(f"agent_code.{candidate}.callbacks")
        original_act = callbacks.act

        def diagnostic_act(agent, game_state):
            action = original_act(agent, game_state)
            callbacks._diagnostic_decision = {
                "round": int(game_state["round"]), "step": int(game_state["step"]),
                "position": list(map(int, game_state["self"][-1])),
                "others": [list(map(int, other[-1])) for other in game_state["others"]],
                "bombs": [list(map(int, position)) for position, _ in game_state["bombs"]],
                "bomb_available": bool(game_state["self"][2]), "action": action,
                "legal_mask": agent.feature_extractor.legal_action_mask(game_state).tolist(),
                "safe_mask": agent.feature_extractor.safe_action_mask(game_state).tolist(),
            }
            return action

        callbacks.act = diagnostic_act

        original_perform = world_class.perform_agent_action

        def diagnostic_perform(world, agent, action):
            if agent.name == "candidate":
                x, y = int(agent.x), int(agent.y)
                deltas = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
                reason = None
                target = None
                if action in deltas:
                    dx, dy = deltas[action]
                    target = (x + dx, y + dy)
                    tile = int(world.arena[target])
                    if tile == -1:
                        reason = "stone_wall"
                    elif tile == 1:
                        reason = "crate"
                    elif any((int(bomb.x), int(bomb.y)) == target for bomb in world.bombs):
                        reason = "bomb_occupied"
                    elif any(other is not agent and (int(other.x), int(other.y)) == target
                             for other in world.active_agents):
                        decision = getattr(callbacks, "_diagnostic_decision", {})
                        snapshot_others = {tuple(position) for position in decision.get("others", [])}
                        reason = ("opponent_already_in_target_at_decision" if target in snapshot_others
                                  else "opponent_moved_into_target_this_turn")
                elif action == "BOMB" and not agent.bombs_left:
                    reason = "bomb_unavailable"
                elif action not in ("WAIT", "BOMB"):
                    reason = "unknown_action"
                if reason is not None:
                    diagnostic_rows.append({
                        "lineup": current["label"], "round": int(world.round),
                        "step": int(world.step), "action": action,
                        "position_at_execution": [x, y],
                        "target": list(target) if target is not None else None,
                        "reason": reason,
                        "decision": getattr(callbacks, "_diagnostic_decision", None),
                        "execution_order": [world.active_agents[int(i)].name
                                            for i in world.replay["permutations"][-1]],
                    })
            return original_perform(world, agent, action)

        world_class.perform_agent_action = diagnostic_perform

    current = {"base": seed, "label": "", "initial": []}
    original_setup_agents = world_class.setup_agents
    original_new_round = world_class.new_round

    def setup_agents(world, agents):
        original_setup_agents(world, agents)
        world.agents[0].name = "candidate"
        world.agents[0].display_name = "candidate"

    def new_round(world):
        round_seed = current["base"] + world.round
        world.rng = np.random.default_rng(round_seed)
        random.seed(round_seed)
        np.random.seed(round_seed)
        for index, agent in enumerate(world.agents):
            runtime = agent.backend.runner.fake_self
            if hasattr(runtime, "rng"):
                runtime.rng.seed(round_seed + index)
        original_new_round(world)
        state = {"field": world.arena.tolist(),
                 "coins": [list(map(int, coin.get_state())) for coin in world.coins],
                 "positions": [[int(agent.x), int(agent.y)] for agent in world.agents]}
        current["initial"].append({"seed": round_seed, "sha256": hashlib.sha256(
            json.dumps(state, sort_keys=True).encode()).hexdigest()})

    world_class.setup_agents = setup_agents
    world_class.new_round = new_round
    summaries = []
    for index, (label, opponents) in enumerate(VALIDATION_LINEUPS):
        current.update(base=seed + index * 1000, label=label, initial=[])
        output = arena / "results" / f"validation_{tag}_{label}.json"
        if output.exists():
            raise FileExistsError(output)
        main.main(["play", "--no-gui", "--agents", candidate, *opponents,
                   "--scenario", "classic", "--n-rounds", str(rounds),
                   "--seed", str(current["base"]), "--save-stats", str(output),
                   "--match-name", f"validation_{tag}_{label}"])
        payload = json.loads(output.read_text(encoding="utf-8"))
        payload["experiment"] = {"training": False, "tag": tag,
                                 "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
                                 "opponents": list(opponents), "initial_conditions": current["initial"]}
        write_json(output, payload)
        summary = next(row for row in summarize_match([output]) if row["agent"] == "candidate")
        summaries.append({"lineup": label, **summary})
        print("VALIDATION", tag, label, json.dumps(summary, sort_keys=True), flush=True)
        close_loggers()
    write_json(arena / "results" / f"validation_{tag}_summary.json", summaries)
    if diagnose_invalid:
        counts = {}
        by_lineup = {}
        for row in diagnostic_rows:
            counts[row["reason"]] = counts.get(row["reason"], 0) + 1
            lineup_counts = by_lineup.setdefault(row["lineup"], {})
            lineup_counts[row["reason"]] = lineup_counts.get(row["reason"], 0) + 1
        write_json(arena / "results" / f"validation_{tag}_invalid_diagnosis.json", {
            "checkpoint": str(checkpoint), "rounds_per_lineup": rounds,
            "counts": counts, "by_lineup": by_lineup, "instances": diagnostic_rows,
        })
        print("INVALID_DIAGNOSIS", tag, json.dumps(counts, sort_keys=True), flush=True)


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--arena", type=Path, required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--arena", type=Path, required=True)
    train_parser.add_argument("--max-blocks", type=int,
                              help="smoke-test only; formal training omits this option")
    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--arena", type=Path, required=True)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--tag", required=True)
    eval_parser.add_argument("--rounds", type=int, default=20)
    eval_parser.add_argument("--seed", type=int, default=VALIDATION_SEED)
    eval_parser.add_argument("--diagnose-invalid", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.arena.resolve())
    elif args.command == "train":
        train(args.arena.resolve(), args.max_blocks)
    else:
        evaluate(args.arena.resolve(), args.checkpoint.resolve(), args.tag, args.rounds,
                 args.seed, args.diagnose_invalid)


if __name__ == "__main__":
    main_cli()
