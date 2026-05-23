# 05 - Imitation Learning

Imitation learning trains a policy to copy a teacher.

Teacher:

- Current strong heuristic agent.
- Filtered successful replays.

Dataset:

- Observation features.
- Teacher action.
- Optional validity mask.

Deployment:

- Neural policy suggests an action.
- Safety filter rejects unsafe choices.
- Fallback to heuristic if model confidence or validity is poor.

