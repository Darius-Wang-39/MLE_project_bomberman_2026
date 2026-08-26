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
- an explicit safe-productive-bomb feature for legal, escapable bombs that can
  destroy a crate;
- a `+0.30` training reward for safe productive bombs and a `-0.20` recent-tile
  revisit penalty when the agent could safely open a nearby crate instead of
  oscillating;
- a contextual training penalty for bombs with neither resource yield nor
  meaningful opponent pressure;
- a `2.5` suicide-penalty multiplier for future training, making the
  `KILLED_SELF` reward `-50` while leaving ordinary `GOT_KILLED` at `-15`.

The bundled checkpoint is the Day 16 rule-retention stage trained with the
current `2.5` multiplier. It reduced the observed fixed-seed midgame alternating
loop from 16 moves to 2 while scoring 132 versus the prior checkpoint's 109 in
the same 30-round rule-agent comparison. Its suicide count was 23 versus 20, so
the loop correction is promoted with that remaining weakness documented.

Training example:

```powershell
python main.py play --no-gui --agents dueling_ddqn_agent --train 1 --scenario coin-heaven --n-rounds 1000
```

The checkpoint `dueling_ddqn_model.pt` contains state dictionaries and counters,
not a pickled model object, so CPU loading remains portable.
