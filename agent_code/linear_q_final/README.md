# Final controlled linear-Q experiment

This experiment adds eight Task-4 interaction features to the original
17-feature linear baseline and compares alpha in `{0.003, 0.001}` with n-step
returns in `{3, 5, 8}`. Each candidate was warm-started from the same 61,973
update checkpoint and trained for 200 classic rounds against three rule agents.

The selected candidate is `frozen_linear_q_final` (`alpha=0.003`, `n_step=8`).
It improved safety but did not pass the declared Task-4 score threshold; see
`notebooks/06_linear_q_final_stopline.ipynb`.
