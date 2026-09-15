# PPO large-scale curriculum branch

Warm-started from `frozen_ppo_task1`. Training seeds are disjoint from all
stage-validation seeds. Each stage is frozen before evaluation; the final
candidate is selected by held-out capability retention rather than recency.
