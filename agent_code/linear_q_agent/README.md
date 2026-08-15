# Linear Q Agent

This is the fast, interpretable baseline model.

## Representation

For every legal action, the agent computes measurements rather than a
hand-written best-action label:

- future escape space;
- present danger and danger reduction;
- change in path distance to coins, useful crate-adjacent tiles, and opponents;
- expected crate/opponent yield when dropping a bomb;
- local freedom and bomb availability.

All directional actions use the same weights, so the representation shares
experience across all eight rotations/reflections of the square board without
duplicating a transition.

## Training

```powershell
python main.py play --no-gui --agents linear_q_agent --train 1 --scenario coin-heaven --n-rounds 1000
```

Training uses five-step returns so delayed crate and coin rewards are credited
back to the bomb placement that caused them. The model is stored as
`linear_q_model.npz` in this directory. Delete that file only when intentionally
starting training from scratch.
