# Imitation from Heuristic

Goal: train a small policy to imitate the strong heuristic agent.

Planned steps:

1. Generate local replay logs.
2. Extract observation-action pairs.
3. Filter emergency or known-bad decisions.
4. Train a supervised classifier over six actions.
5. Evaluate behind the safety filter.

No training is implemented yet.

