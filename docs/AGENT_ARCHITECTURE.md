# Bomberland Agent Architecture

## 1. Overview

The current production direction is a hybrid tactical agent: rule-based decisions, BFS search, danger-map reasoning, and heuristic scoring. There is no neural model in production yet. The goal is to keep hard safety constraints deterministic while gradually adding long-term value estimates.

## 2. Core Components

- State parser: reads `map`, `players`, and `bombs` from the observation.
- Danger map: estimates when each cell becomes unsafe from bomb blasts and chain reactions.
- Safety layer: rejects immediate-death moves and prioritizes escape when the current cell is dangerous.
- BFS pathfinding: finds safe cells, items, box-bomb spots, and enemy approach routes.
- Bomb escape validation: `PLACE_BOMB` is valid only when escape after placement is possible.
- Future survivability: scores reachable safe space after a candidate move.
- Enemy escape pressure: estimates whether a bomb reduces enemy safe reachable cells.
- Loop breaker: discourages local repeated positions and passive STOP loops.
- Controlled expansion: rewards safe map opening, branch points, items, boxes, and center progress.
- Territory pressure: adds light value for occupying safe open space that constrains enemies.
- Bomb policy: bombs must be meaningful for boxes, direct enemy threat, or escape pressure.

## 3. Safety Invariants

- Never select an action whose next cell has `danger_time <= 1`.
- `PLACE_BOMB` always requires `can_escape_after_bomb`.
- Bombs should be meaningful; avoid bomb spam.
- No network, API calls, LLM calls, subprocesses, or disk writes inside `act()`.
- `act()` must stay under 100 ms.

## 4. Online Robustness Philosophy

Optimize for online leaderboard behavior, not local-only score. Local baselines are useful, but hidden online opponents can punish overfitted aggression. The failed less-draw direction showed that reducing draws by forcing aggression can lower online win rate and increase average steps. Prefer stable territory value, safe expansion, and conservative pressure.

## 5. Future Research Extensions

- Imitation learning from the heuristic agent.
- Tiny policy head that suggests actions behind the rule-based safety filter.
- Tiny Q-head for value estimates, also behind the safety filter.
- PPO only after a stable dataset, stable simulator pipeline, and action masking exist.

