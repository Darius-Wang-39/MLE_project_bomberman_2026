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
- chain-reaction-aware bomb timing and static opponent blocking in the six-step
  survival planner;
- a hard two-branch escape requirement before inference, exploration, or a
  Double-DQN target may select `BOMB`;
- a recent-position Q penalty, active only when robust bomb safety rejects
  `BOMB`, so the agent changes route instead of oscillating in a narrow area;
- a `+0.30` training reward for safe productive bombs and a `-0.20` recent-tile
  revisit penalty when the agent could safely open a nearby crate instead of
  oscillating;
- a contextual training penalty for bombs with neither resource yield nor
  meaningful opponent pressure;
- a `2.5` suicide-penalty multiplier for future training, making the
  `KILLED_SELF` reward `-50` while leaving ordinary `GOT_KILLED` at `-15`.

The bundled checkpoint is the Day 17 robust-safety stage trained with the
current `2.5` multiplier. The strict safety-only evaluation reached 0.26
suicides and 2.24 score per round. The final balanced controller reached 0.34
suicides, 2.70 score, and 0.30 kills per round over 50 rule-agent rounds; on the
visual regression seed, its longest opposite-direction loop was 5 moves rather
than the safety-only controller's 75.

Training example:

```powershell
python main.py play --no-gui --agents dueling_ddqn_agent --train 1 --scenario coin-heaven --n-rounds 1000
```

The checkpoint `dueling_ddqn_model.pt` contains state dictionaries and counters,
not a pickled model object, so CPU loading remains portable.
