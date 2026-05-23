# Bomberland Research Workspace

This folder is for ML/DL/RL exploration that must not affect the production agent until an experiment is validated.

Rules:

- Do not overwrite `agent/hybrid_agent/agent.py` from research code.
- Start with analysis and datasets before model training.
- Prefer imitation learning before PPO.
- Keep all learned policies behind a deterministic safety filter.
- Record results in `docs/EXPERIMENT_LOG.md`.

Recommended path:

1. Study replay failures.
2. Collect heuristic obs-action pairs.
3. Train a small imitation policy.
4. Evaluate with action masks and safety veto.
5. Only then consider DQN/PPO experiments.

