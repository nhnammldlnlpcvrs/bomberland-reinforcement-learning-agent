# Reward and Heuristic Design for Bomberland

## 1. RL Concepts

- Reward: immediate signal after an action.
- Return: discounted future reward.
- Value: expected return from a state.
- `Q(s,a)`: expected return after taking action `a` in state `s`.
- Advantage: how much better an action is than the average action in that state.
- GAE: a practical advantage estimator for actor-critic methods.
- Reward shaping risk: extra rewards can produce unintended behavior.

## 2. Bomberland Reward Candidates

- Survival
- Win
- Kill
- Box destruction
- Item pickup
- Territory expansion
- Enemy space denial
- Safe reachable cells
- Bomb escape margin
- Stuck-loop penalty
- Self-bomb penalty

## 3. Heuristic Mapping

| Reward Idea | Current Heuristic Equivalent |
|---|---|
| survival | danger map, safe action filter |
| future value | future survivability and expansion score |
| kill pressure | enemy escape pressure |
| territory | controlled expansion and center progress |
| anti-loop | position history repeat penalty |
| bomb safety | can_escape_after_bomb |

## 4. Bad Reward Shaping

- Over-aggression that lowers online win rate.
- Local-only score chasing.
- Bomb spam that reduces mobility.
- Camping that increases draw rate.

## 5. Recommended Direction

Prioritize long-term territory value, controlled expansion, and a safe action filter. Let learned models rank safe choices; do not let them bypass safety.

