# 02 - Value and Q-learning

Value estimates the long-term quality of a state. Q-value estimates the long-term quality of an action from a state.

For Bomberland, useful Q-style features include:

- Is the next cell safe now and later?
- How many safe cells are reachable after moving?
- Does this move open more territory?
- Does a bomb reduce enemy escape options?

Do not let Q-values bypass safety. They should rank valid actions only.

