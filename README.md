Frozen Lake Q-Learning Agent
A  reinforcement-learning pipeline that trains an agent via tabular Q-learning to solve Gymnasium’s FrozenLake-v1 environment on the 8 × 8 map.

Key Features
Advanced ε-greedy Q-learning with dynamic ε-decay and learning-rate scheduling.

State persistence – automatically serialises and reloads Q-tables (.pkl).

Performance analytics – saves a rolling-average reward plot for quick visual insight.

Head-less or rendered execution – toggle real-time visuals with one flag.

Modular entry point – run full training, inference-only, or mixed sessions in a single call.
