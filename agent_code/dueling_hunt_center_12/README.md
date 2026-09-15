# Hunt-center inference ablation (0.12)

This frozen inference-only variant uses the exact checkpoint, feature extractor,
network, safety mask, revisit penalty, and objective bias from
\`dueling_ddqn_agent\`. Its only policy parameter change is:

\`\`\`python
HUNT_CENTER_Q_BIAS = 0.12
\`\`\`

The bias is applied only in the hunt phase (no coins, no crates, at least one
opponent). Do not train this directory.
