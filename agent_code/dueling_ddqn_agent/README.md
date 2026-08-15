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
- safety and legality masks during exploration, training targets, and inference.

Training example:

```powershell
python main.py play --no-gui --agents dueling_ddqn_agent --train 1 --scenario coin-heaven --n-rounds 1000
```

The checkpoint `dueling_ddqn_model.pt` contains state dictionaries and counters,
not a pickled model object, so CPU loading remains portable.

