# 01 - MDP for Bomberland

An MDP has state, action, transition, reward, and horizon.

In Bomberland:

- State: board map, players, bombs, derived danger map.
- Action: 0-5 discrete action.
- Transition: engine step after all agents act.
- Reward: win/loss/draw plus optional shaping.
- Horizon: up to 500 steps.

Practical note: the observation is fully visible, but the opponent policies are unknown. Treat online evaluation as a different opponent distribution.

