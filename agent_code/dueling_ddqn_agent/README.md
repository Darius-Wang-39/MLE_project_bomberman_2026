# Dueling Double DQN Agent

Competition-oriented model using the same teacher-compliant action measurements
as the linear baseline, but learning nonlinear interactions between them.

Implemented components:

- Dueling value/advantage network;
- true Double DQN targets;
- five-step returns;
- proportional prioritized replay;
- importance-sampling weights;
- Huber loss and gradient clipping;
- soft target-network updates;
- one randomly selected D4 symmetry per sampled transition;
- safety and legality masks during exploration, training targets, and inference;
- a soft `0.08` Q-value bias toward the central inclusive 11x11 box during the
  target-free opponent-hunting endgame;
- opponent escape-pressure and armed-opponent threat features;
- a contextual training penalty for bombs with neither resource yield nor
  meaningful opponent pressure;
- a `3.0` suicide-penalty multiplier, making the `KILLED_SELF` training reward
  `-60` while leaving ordinary `GOT_KILLED` at `-15`.

Training example:

```powershell
python main.py play --no-gui --agents dueling_ddqn_agent --train 1 --scenario coin-heaven --n-rounds 1000
```

The checkpoint `dueling_ddqn_model.pt` contains state dictionaries and counters,
not a pickled model object, so CPU loading remains portable.
