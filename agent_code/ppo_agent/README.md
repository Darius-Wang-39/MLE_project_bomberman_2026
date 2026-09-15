# PPO agent

Independent masked actor-critic PPO using the current DDQN action-relative
features, safe-action mask, and reward shaping. It uses fresh on-policy
rollouts, GAE, clipped updates, deterministic evaluation, and no replay buffer.
