# Rule demonstration collector

This development-only agent delegates actions to the supplied
`rule_based_agent` and stores the DDQN feature vector, teacher action, safe-action
mask, round, and step. It is never part of the tournament submission.

Set `BOMBERMAN_DEMO_PATH` and run it as the first training agent so the
`end_of_round` hook flushes the dataset:

```bash
BOMBERMAN_DEMO_PATH=experiments/demonstrations/rule_classic.npz \
python main.py play --no-gui --train 1 \
  --agents demo_rule_agent rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic --n-rounds 50 --seed 5101
```

