# 04 - PPO Notes

PPO is an actor-critic method with clipped policy updates. It is powerful but not the first step for this repo.

Use PPO only when:

- Replay logging is stable.
- Reward shaping is tested.
- Action masks are implemented.
- A supervised or heuristic baseline exists.
- Self-play curriculum is available.

Bomberland PPO should keep a rule-based safety veto in deployment.

