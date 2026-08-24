# Behaviour-cloning evaluation adapter

This development-only agent loads the Day 5 behaviour-cloned checkpoint while
reusing the competition candidate's exact inference callbacks. It keeps the
offline warm-start ablation separate from the RL-trained checkpoint.

