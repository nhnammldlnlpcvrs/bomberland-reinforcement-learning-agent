# 03 - DQN Notes

DQN learns Q-values from replayed transitions. For Bomberland, a tiny DQN could use:

- 13x13 multi-channel board tensor.
- Optional scalar features for bombs left, radius bonus, enemy distances.
- Six Q-values as output.

Risks:

- Reward shaping can teach bomb spam.
- Local baselines may overfit.
- Unsafe actions must be masked before action selection.

Start with a tiny model and compare against the heuristic teacher.

