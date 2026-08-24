"""Pretrain a fresh Dueling DDQN network from rule-agent demonstrations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_code.dueling_ddqn_agent.callbacks import INPUT_DIM
from agent_code.dueling_ddqn_agent.features import ACTIONS, FEATURE_NAMES
from agent_code.dueling_ddqn_agent.model import DuelingQNetwork
from agent_code.dueling_ddqn_agent.symmetry import (
    D4_TRANSFORMS,
    transform_action_features,
    transform_action_index,
)


def load_demonstrations(path: Path):
    with np.load(path, allow_pickle=False) as data:
        states = np.asarray(data["states"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.int64)
        safe_masks = np.asarray(data["safe_masks"], dtype=bool)
        rounds = np.asarray(data["rounds"], dtype=np.int32)
    if states.ndim != 2 or states.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected states shaped (N, {INPUT_DIM}), got {states.shape}")
    if not (len(states) == len(actions) == len(safe_masks) == len(rounds)):
        raise ValueError("Demonstration arrays have inconsistent lengths")
    return states, actions, safe_masks, rounds


def split_rounds(rounds: np.ndarray, validation_fraction: float, seed: int):
    unique_rounds = np.unique(rounds)
    if unique_rounds.size < 2:
        raise ValueError("At least two rounds are required for a leakage-free split")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_rounds)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_rounds = shuffled[:validation_count]
    validation_mask = np.isin(rounds, validation_rounds)
    return ~validation_mask, validation_mask


def augment_batch(states: np.ndarray, actions: np.ndarray, rng: np.random.Generator):
    augmented_states = np.empty_like(states)
    augmented_actions = np.empty_like(actions)
    for index in range(len(states)):
        transform = D4_TRANSFORMS[int(rng.integers(len(D4_TRANSFORMS)))]
        augmented_states[index] = transform_action_features(
            states[index], transform, len(FEATURE_NAMES)
        )
        augmented_actions[index] = transform_action_index(int(actions[index]), transform)
    return augmented_states, augmented_actions


def accuracy(network, states: np.ndarray, actions: np.ndarray, device, batch_size: int):
    network.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch = torch.as_tensor(states[start : start + batch_size], device=device)
            predictions = network(batch).argmax(dim=1).cpu().numpy()
            correct += int(np.sum(predictions == actions[start : start + batch_size]))
    return correct / max(len(states), 1)


def pretrain(
    dataset: Path,
    output: Path,
    epochs: int = 15,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    seed: int = 6101,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    states, actions, safe_masks, rounds = load_demonstrations(dataset)
    train_mask, validation_mask = split_rounds(rounds, 0.2, seed)
    train_states, train_actions = states[train_mask], actions[train_mask]
    validation_states, validation_actions = states[validation_mask], actions[validation_mask]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = DuelingQNetwork(INPUT_DIM, len(ACTIONS)).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)

    history = []
    for epoch in range(1, epochs + 1):
        network.train()
        permutation = rng.permutation(len(train_states))
        total_loss = 0.0
        seen = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            batch_states, batch_actions = augment_batch(
                train_states[indices], train_actions[indices], rng
            )
            states_t = torch.as_tensor(batch_states, device=device)
            actions_t = torch.as_tensor(batch_actions, device=device)
            loss = criterion(network(states_t), actions_t)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 10.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
            seen += len(indices)
        validation_accuracy = accuracy(
            network, validation_states, validation_actions, device, batch_size
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(seen, 1),
                "validation_accuracy": validation_accuracy,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "policy_state": {key: value.detach().cpu() for key, value in network.state_dict().items()},
        "optimizer_state": None,
        "environment_steps": 0,
        "learner_updates": 0,
        "pretraining_epochs": epochs,
        "demonstration_count": len(states),
        "validation_accuracy": history[-1]["validation_accuracy"],
    }
    torch.save(checkpoint, output)
    unsafe_teacher_fraction = float(
        np.mean(~safe_masks[np.arange(len(actions)), actions])
    )
    return {
        "dataset": str(dataset),
        "output": str(output),
        "samples": len(states),
        "rounds": int(len(np.unique(rounds))),
        "train_samples": int(np.sum(train_mask)),
        "validation_samples": int(np.sum(validation_mask)),
        "unsafe_teacher_fraction": unsafe_teacher_fraction,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = pretrain(
        args.dataset,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
