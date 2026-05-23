# ML Pipeline Skill

Use this skill when adding ML/DL/RL infrastructure.

Preferred roadmap:

1. Collect datasets from heuristic agents and replay logs.
2. Train imitation learning first.
3. Evaluate with safety filter and action masks.
4. Try tiny DQN/Q-head only after dataset quality is known.
5. Use PPO only after a stable supervised baseline and reward design.

Constraints:

- Keep models small enough for 100 ms `act()`.
- Do not add dependencies without explicit approval.
- Do not train models inside ordinary repo-organization tasks.
- Do not commit large datasets or checkpoints.
- Keep production submission separate from research code.

